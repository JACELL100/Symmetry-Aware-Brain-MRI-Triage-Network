import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from config import CFG
from dataset import build_loaders
from model import SymmetryAwareTriageNet
from train import train_one_epoch, evaluate


def set_seed(seed):
    """
    Make training more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training():
    """
    Full training and evaluation routine.
    """

    cfg = CFG()
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader, class_names, class_to_idx = build_loaders(cfg)

    print("Class order:", class_names)
    print("Class mapping:", class_to_idx)
    print("Device:", cfg.device)

    model = SymmetryAwareTriageNet(
        num_classes=len(class_names),
        dropout=cfg.dropout,
        use_imagenet_weights=cfg.use_imagenet_weights,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Start below any possible F1 so the first epoch always saves a checkpoint.
    best_f1 = -1.0
    save_path = "best_kaggle_symmetry_triage_model.pth"

    for epoch in range(cfg.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            cfg=cfg,
        )

        val_loss, val_acc, val_f1, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            cfg=cfg,
            class_names=class_names,
        )

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Save best model according to validation macro F1.
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "class_to_idx": class_to_idx,
                    "cfg": {
                        "img_size": cfg.img_size,
                        "dropout": cfg.dropout,
                        "use_imagenet_weights": cfg.use_imagenet_weights,
                        "confidence_threshold": cfg.confidence_threshold,
                        "entropy_threshold": cfg.entropy_threshold,
                        "mc_samples": cfg.mc_samples,
                    },
                },
                save_path,
            )

    # Load best checkpoint before final testing.
    checkpoint = torch.load(save_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc, test_f1, test_report = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        cfg=cfg,
        class_names=class_names,
    )

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(test_report)


if __name__ == "__main__":
    run_training()
