import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_one_epoch(model, loader, optimizer, criterion, cfg):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        optimizer.zero_grad()

        # Forward pass.
        logits, _ = model(images)
        loss = criterion(logits, labels)

        # Backpropagation.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, class_names=None):
    """
    Evaluate model without gradient computation.
    """

    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    report = None
    if class_names is not None:
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

    return total_loss / len(loader), acc, f1, report
