from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download


# Hugging Face repo info
HF_REPO_ID = "mudz12345/seaweed-convnext-tiny"
HF_FILENAME = "convnext_tiny_best_checkpoint.pth"

# Your pasted URL is a browser "blob" URL.
# For backend loading, use huggingface_hub with repo_id + filename instead.
HF_BLOB_URL = "https://huggingface.co/mudz12345/seaweed-convnext-tiny/blob/main/convnext_tiny_best_checkpoint.pth"

BEST_MODEL_NAME = "convnext_tiny"
CLASS_NAMES = ["healthy", "unhealthy"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_img_size(model_name: str, num_classes: int = 2):
    if model_name != "convnext_tiny":
        raise ValueError(f"Unsupported model in this backend: {model_name}")

    model = models.convnext_tiny(weights=None)
    img_size = 224

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model, img_size


def download_model_from_hf() -> Path:
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
    )
    return Path(local_path)


def load_model_for_inference():
    weights_path = download_model_from_hf()

    model, img_size = get_model_and_img_size(
        model_name=BEST_MODEL_NAME,
        num_classes=len(CLASS_NAMES),
    )

    checkpoint = torch.load(weights_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)

    return model, img_size


best_model, best_img_size = load_model_for_inference()