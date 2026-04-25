import torch
import torch.nn.functional as F


def enable_dropout(model):
    """
    Activate Dropout layers during inference.

    This is required for MC Dropout uncertainty estimation.
    """

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model, image, cfg):
    """
    Run multiple stochastic predictions for a single image.

    image shape: [1, C, H, W]
    """

    model.eval()
    enable_dropout(model)

    probs_list = []
    attention_maps = []

    for _ in range(cfg.mc_samples):
        logits, attention_map = model(image.to(cfg.device))
        probs = F.softmax(logits, dim=1)

        probs_list.append(probs)
        attention_maps.append(attention_map)

    # Shape: [T, 1, num_classes]
    probs_stack = torch.stack(probs_list, dim=0)

    # Mean class probability across stochastic passes.
    mean_probs = probs_stack.mean(dim=0)

    # Average variance across classes, used as a rough uncertainty signal.
    variance = probs_stack.var(dim=0).mean().item()

    confidence, pred_class = mean_probs.max(dim=1)

    # Predictive entropy. Higher entropy means less certain prediction.
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)

    return {
        "pred_class": pred_class.item(),
        "confidence": confidence.item(),
        "entropy": entropy.item(),
        "variance": variance,
        "attention_map": attention_maps[-1],
        "mean_probs": mean_probs.squeeze(0).cpu(),
    }
