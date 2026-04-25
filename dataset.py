from PIL import ImageOps
import numpy as np

from sklearn.model_selection import train_test_split

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class CropBlackMargins:
    """
    Crop dark borders around MRI images.

    Many images in the Kaggle dataset contain black margins.
    Removing these margins helps the model focus on the brain region
    rather than background pixels.
    """

    def __init__(self, threshold=10):
        # Pixels brighter than this threshold are considered foreground.
        self.threshold = threshold

    def __call__(self, img):
        # Convert to grayscale so border detection depends on intensity only.
        gray = ImageOps.grayscale(img)
        arr = np.array(gray)

        # Foreground mask: non-black pixels.
        mask = arr > self.threshold

        # If the image is empty or fully dark, return a safe RGB version.
        if not mask.any():
            return gray.convert("RGB")

        # Locate bounding box of foreground pixels.
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        # Crop to foreground and convert back to 3 channels for ResNet.
        cropped = gray.crop((x0, y0, x1, y1))
        return cropped.convert("RGB")


def validate_kaggle_structure(cfg):
    """
    Ensure the local dataset matches the Kaggle folder structure.
    This catches path mistakes before training starts.
    """

    required_dirs = [
        cfg.train_dir / "glioma",
        cfg.train_dir / "meningioma",
        cfg.train_dir / "pituitary",
        cfg.train_dir / "notumor",
        cfg.test_dir / "glioma",
        cfg.test_dir / "meningioma",
        cfg.test_dir / "pituitary",
        cfg.test_dir / "notumor",
    ]

    missing = [str(path) for path in required_dirs if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Dataset structure is incorrect. Missing folders:\n"
            + "\n".join(missing)
        )


def build_transforms(cfg):
    """
    Build train and evaluation transforms.

    Horizontal flipping is intentionally excluded because left-right
    orientation is meaningful for symmetry-aware attention.
    """

    train_tfms = transforms.Compose([
        CropBlackMargins(threshold=10),
        transforms.Resize((cfg.img_size, cfg.img_size)),

        # Mild augmentations that preserve rough anatomical layout.
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),

        transforms.ToTensor(),

        # ImageNet normalization matches the pretrained ResNet-18 backbone.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        CropBlackMargins(threshold=10),
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, eval_tfms


def stratified_train_val_split(dataset, val_split, seed):
    """
    Create a class-balanced validation split from the Training folder.
    The Testing folder remains untouched for final evaluation.
    """

    targets = [label for _, label in dataset.samples]
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=targets,
    )

    return train_idx.tolist(), val_idx.tolist()


def build_loaders(cfg):
    """
    Return train, validation, and test DataLoaders plus class metadata.
    """

    validate_kaggle_structure(cfg)

    train_tfms, eval_tfms = build_transforms(cfg)

    # Same Training directory is loaded twice so train and val can use
    # different transforms while sharing the same image files.
    full_train_ds = datasets.ImageFolder(
        root=str(cfg.train_dir),
        transform=train_tfms,
    )

    full_val_ds = datasets.ImageFolder(
        root=str(cfg.train_dir),
        transform=eval_tfms,
    )

    # Testing is used only after model selection.
    test_ds = datasets.ImageFolder(
        root=str(cfg.test_dir),
        transform=eval_tfms,
    )

    train_idx, val_idx = stratified_train_val_split(
        dataset=full_train_ds,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_val_ds, val_idx)

    pin_memory = cfg.device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        full_train_ds.classes,
        full_train_ds.class_to_idx,
    )
