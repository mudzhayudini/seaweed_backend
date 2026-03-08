from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam, class_idx


def get_eval_transform_only(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def overlay_cam_on_image(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.4):
    img = np.array(pil_img).astype(np.float32) / 255.0
    h, w = img.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    return cam_resized, heatmap, overlay


def summarize_gradcam(cam_resized: np.ndarray, threshold: float = 0.6) -> Dict[str, Any]:
    h, w = cam_resized.shape
    mask = cam_resized >= threshold

    hotspot_pixels = int(mask.sum())
    hotspot_fraction = hotspot_pixels / float(h * w)

    if hotspot_pixels == 0:
        return {
            "threshold": threshold,
            "hotspot_fraction": 0.0,
            "bbox_norm": None,
            "centroid_norm": None,
            "region_description": "No strong hotspot above threshold.",
        }

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h

    bbox_norm = {
        "x_min": float(x_min) / w,
        "y_min": float(y_min) / h,
        "x_max": float(x_max) / w,
        "y_max": float(y_max) / h,
    }

    horiz = "left" if cx < 0.33 else "center" if cx < 0.67 else "right"
    vert = "upper" if cy < 0.33 else "middle" if cy < 0.67 else "lower"
    region_description = f"Primary attention is concentrated in the {vert}-{horiz} region of the image."

    return {
        "threshold": threshold,
        "hotspot_fraction": float(hotspot_fraction),
        "bbox_norm": bbox_norm,
        "centroid_norm": {"x": float(cx), "y": float(cy)},
        "region_description": region_description,
    }