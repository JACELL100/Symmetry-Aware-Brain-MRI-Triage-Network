import torch
from PIL import Image
from torchvision import transforms

from config import CFG
from dataset import CropBlackMargins
from model import SymmetryAwareTriageNet
from uncertainty import mc_dropout_predict
from triage import format_output


def build_inference_transform(cfg):
    """
    Build the exact preprocessing pipeline used for evaluation.
    """

    return transforms.Compose([
        CropBlackMargins(threshold=10),
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image(image_path, cfg):
    """
    Load one image from local drive and convert it into a model input tensor.
    """

    image = Image.open(image_path).convert("RGB")
    transform = build_inference_transform(cfg)
    return transform(image).unsqueeze(0)


def apply_saved_cfg(cfg, saved_cfg):
    """
    Restore critical training-time configuration values.

    This prevents inference from accidentally using a different image size,
    dropout value, or uncertainty threshold from the trained checkpoint.
    """

    cfg.img_size = saved_cfg.get("img_size", cfg.img_size)
    cfg.dropout = saved_cfg.get("dropout", cfg.dropout)
    cfg.use_imagenet_weights = saved_cfg.get(
        "use_imagenet_weights",
        cfg.use_imagenet_weights,
    )
    cfg.confidence_threshold = saved_cfg.get(
        "confidence_threshold",
        cfg.confidence_threshold,
    )
    cfg.entropy_threshold = saved_cfg.get(
        "entropy_threshold",
        cfg.entropy_threshold,
    )
    cfg.mc_samples = saved_cfg.get("mc_samples", cfg.mc_samples)

    return cfg


def predict_single_image(image_path, model_path):
    """
    Predict one image and return class plus uncertainty-aware triage.
    """

    cfg = CFG()

    checkpoint = torch.load(model_path, map_location=cfg.device)
    cfg = apply_saved_cfg(cfg, checkpoint.get("cfg", {}))

    class_names = checkpoint["class_names"]

    model = SymmetryAwareTriageNet(
        num_classes=len(class_names),
        dropout=cfg.dropout,
        use_imagenet_weights=cfg.use_imagenet_weights,
    ).to(cfg.device)

    model.load_state_dict(checkpoint["model_state"])

    image = load_image(image_path, cfg)

    prediction = mc_dropout_predict(model, image, cfg)
    output = format_output(prediction, class_names, cfg)

    return output


if __name__ == "__main__":
    result = predict_single_image(
        image_path=r"D:\datasets\brain-tumor-mri-dataset\Testing\glioma\Te-gl_0010.jpg",
        model_path="best_kaggle_symmetry_triage_model.pth",
    )

    print(result)
