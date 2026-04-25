import torch
from pathlib import Path


class CFG:
    # Local path to the extracted Kaggle dataset.
    # Change this path to match your machine.
    data_root = Path(r"D:\datasets\brain-tumor-mri-dataset")

    # Kaggle dataset uses Training and Testing folders, not train/val/test.
    train_dir = data_root / "Training"
    test_dir = data_root / "Testing"

    # 256 is used instead of 224 because ResNet produces an even-width feature map.
    # This makes left/right feature splitting cleaner for contralateral attention.
    img_size = 256

    # Training settings.
    batch_size = 32
    num_classes = 4
    epochs = 40
    lr = 1e-4
    weight_decay = 1e-4
    dropout = 0.35

    # Validation split is created only from the Training folder.
    val_split = 0.15
    seed = 42

    # DataLoader workers. Use 0 on Windows if multiprocessing causes issues.
    num_workers = 4

    # Automatically use GPU if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Number of stochastic forward passes for MC Dropout uncertainty.
    mc_samples = 20

    # Triage thresholds. These are placeholders and should be tuned on validation data.
    confidence_threshold = 0.70
    entropy_threshold = 0.65

    # If the machine has no internet and ResNet weights are not cached, set to False.
    use_imagenet_weights = True
