import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetFeatureEncoder(nn.Module):
    """
    ResNet-18 feature extractor.

    The classification head of ResNet is removed. The output is a dense
    feature map used by the symmetry-aware and global branches.
    """

    def __init__(self, use_imagenet_weights=True):
        super().__init__()

        # Use ImageNet weights when available. Set use_imagenet_weights=False
        # in config.py for offline environments without cached weights.
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        backbone = resnet18(weights=weights)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # ResNet-18 layer4 output channels.
        self.out_channels = 512

    def forward(self, x):
        return self.encoder(x)


class ContralateralAttention(nn.Module):
    """
    Center-aligned contralateral attention.

    The feature map is split into left and right halves. Each side attends
    to the mirrored counterpart from the opposite side. A local spatial mask
    restricts attention to nearby mirrored positions, keeping the comparison
    anatomically meaningful instead of fully global.
    """

    def __init__(self, channels, heads=4, dropout=0.1, local_radius=2):
        super().__init__()

        self.local_radius = local_radius

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        # Projection compresses local, contralateral, difference, and interaction
        # features back to the original channel dimension.
        self.proj = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
        )

    def _spatial_mask(self, h, w, device):
        """
        Build a local Manhattan-distance attention mask.

        Positions beyond local_radius receive -inf, so attention cannot select
        them. This keeps each token focused on a small mirrored neighborhood.
        """

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )

        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()
        dist = torch.cdist(coords, coords, p=1)

        mask = torch.zeros_like(dist)
        mask = mask.masked_fill(dist > self.local_radius, float("-inf"))

        return mask

    def _attend(self, query_feat, mirror_feat):
        """
        Apply masked attention from one hemisphere to its mirrored counterpart.

        query_feat:  [B, C, H, W_half]
        mirror_feat: [B, C, H, W_half]
        """

        b, c, h, w = query_feat.shape

        # Convert spatial feature maps into token sequences.
        query_tokens = query_feat.flatten(2).transpose(1, 2)
        mirror_tokens = mirror_feat.flatten(2).transpose(1, 2)

        attn_mask = self._spatial_mask(
            h=h,
            w=w,
            device=query_feat.device,
        )

        # Query = local side, Key/Value = mirrored opposite side.
        context, _ = self.attn(
            query=query_tokens,
            key=mirror_tokens,
            value=mirror_tokens,
            attn_mask=attn_mask,
        )

        # Build asymmetry-aware token descriptors.
        asym_tokens = torch.cat(
            [
                query_tokens,                     # Local appearance
                context,                          # Contralateral context
                torch.abs(query_tokens - context),  # Absolute mismatch
                query_tokens * context,           # Feature interaction
            ],
            dim=-1,
        )

        asym_tokens = self.proj(asym_tokens)

        # Convert tokens back to spatial feature map.
        return asym_tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, feat):
        """
        feat shape: [B, C, H, W]
        """

        _, _, _, w = feat.shape
        mid = w // 2

        # Handle odd-width feature maps safely, although img_size=256 normally
        # produces an even feature width through ResNet-18.
        if w % 2 == 1:
            left = feat[:, :, :, :mid]
            center = feat[:, :, :, mid:mid + 1]
            right = feat[:, :, :, mid + 1:]
        else:
            left = feat[:, :, :, :mid]
            center = None
            right = feat[:, :, :, mid:]

        # Mirror each half so corresponding anatomical directions align.
        right_mirror = torch.flip(right, dims=[-1])
        left_mirror = torch.flip(left, dims=[-1])

        # Compute attention in both directions.
        left_asym = self._attend(left, right_mirror)
        right_asym = self._attend(right, left_mirror)

        # Reassemble full-width feature map.
        if center is not None:
            return torch.cat([left_asym, center, right_asym], dim=-1)

        return torch.cat([left_asym, right_asym], dim=-1)


class GlobalContextBranch(nn.Module):
    """
    Global context branch.

    This branch keeps broad image context so the model does not rely only on
    left-right asymmetry. This is important for central tumors such as
    pituitary lesions.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat):
        return self.block(feat)


class FeatureFusion(nn.Module):
    """
    Fuse asymmetry-aware features with global-context features.
    """

    def __init__(self, channels):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, asym_feat, global_feat):
        fused = torch.cat([asym_feat, global_feat], dim=1)
        return self.fusion(fused)


class InformativeRegionAggregator(nn.Module):
    """
    Attention-based region aggregation.

    Each spatial token receives an importance score. The final image-level
    representation is the weighted sum of all tokens.
    """

    def __init__(self, channels, hidden_dim=256):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        b, c, h, w = feat.shape

        # [B, C, H, W] -> [B, H*W, C]
        tokens = feat.flatten(2).transpose(1, 2)

        # Region importance scores.
        scores = self.attention(tokens)
        weights = torch.softmax(scores, dim=1)

        # Weighted aggregation into one image-level descriptor.
        representation = torch.sum(tokens * weights, dim=1)

        # Keep attention map for interpretability.
        attention_map = weights.transpose(1, 2).reshape(b, 1, h, w)

        return representation, attention_map


class SymmetryAwareTriageNet(nn.Module):
    """
    Full proposed model.

    Output:
      logits: raw class scores
      attention_map: informative-region map for visualization
    """

    def __init__(self, num_classes=4, dropout=0.35, use_imagenet_weights=True):
        super().__init__()

        self.encoder = ResNetFeatureEncoder(
            use_imagenet_weights=use_imagenet_weights,
        )

        channels = self.encoder.out_channels

        self.contralateral_attention = ContralateralAttention(
            channels=channels,
            heads=4,
            dropout=0.1,
            local_radius=2,
        )

        self.global_branch = GlobalContextBranch(channels)
        self.fusion = FeatureFusion(channels)
        self.aggregator = InformativeRegionAggregator(channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Extract dense features from the MRI image.
        feat = self.encoder(x)

        # Build symmetry-aware and global-context representations.
        asym_feat = self.contralateral_attention(feat)
        global_feat = self.global_branch(feat)

        # Fuse both representations and aggregate informative regions.
        fused_feat = self.fusion(asym_feat, global_feat)
        representation, attention_map = self.aggregator(fused_feat)

        # Final class logits.
        logits = self.classifier(representation)

        return logits, attention_map
