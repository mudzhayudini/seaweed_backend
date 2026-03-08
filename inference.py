import io
import json
import base64
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

from model_loader import (
    best_model,
    best_img_size,
    BEST_MODEL_NAME,
    CLASS_NAMES,
    DEVICE,
)
from gradcam_utils import (
    GradCAM,
    get_eval_transform_only,
    overlay_cam_on_image,
    summarize_gradcam,
)
from image_utils import segment_and_crop_seaweed
from deepseek_utils import call_deepseek_api


def numpy_rgb_to_base64_png(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_seaweed_with_best_model(
    image_path,
    true_label=None,
    alpha: float = 0.4,
    gradcam_threshold: float = 0.6,
):
    model = best_model
    img_size = best_img_size
    model_name = BEST_MODEL_NAME

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    eval_tf = get_eval_transform_only(img_size)

    image_path = Path(image_path)
    pil_img_original = Image.open(image_path).convert("RGB")
    pil_img_cropped, seg_mask, segmented_rgb = segment_and_crop_seaweed(
        pil_img_original,
        padding=10,
    )

    pil_img = pil_img_cropped
    input_tensor = eval_tf(pil_img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    with torch.enable_grad():
        cam, _ = gradcam.generate(input_tensor, class_idx=pred_idx)
    gradcam.remove_hooks()

    cam_resized, heatmap, overlay = overlay_cam_on_image(pil_img, cam, alpha=alpha)
    gradcam_summary = summarize_gradcam(cam_resized, threshold=gradcam_threshold)

    img_np = np.array(pil_img)
    image_summary = {
        "file_name": image_path.name,
        "image_size_pixels": {
            "height": int(img_np.shape[0]),
            "width": int(img_np.shape[1]),
        },
        "model_input_size": int(img_size),
    }

    return {
        "model_name": model_name,
        "image_path": str(image_path),
        "original_pil": pil_img_original,
        "cropped_pil": pil_img_cropped,
        "segmented_rgb": segmented_rgb,
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": {
            "healthy": float(probs[0]),
            "unhealthy": float(probs[1]),
        },
        "gradcam_summary": gradcam_summary,
        "image_summary": image_summary,
        "overlay_np": overlay,
        "heatmap_np": heatmap,
        "cam_np": cam_resized,
        "true_label": true_label,
    }


def generate_deepseek_explanation_from_result(result: Dict[str, Any]) -> str:
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    gradcam_summary = result["gradcam_summary"]
    image_summary = result["image_summary"]
    true_label = result.get("true_label", None)
    model_name = result["model_name"]

    confidence_pct = confidence * 100
    healthy_pct = probs["healthy"] * 100
    unhealthy_pct = probs["unhealthy"] * 100
    is_uncertain = abs(probs["healthy"] - probs["unhealthy"]) < 0.15

    system_message = (
        "You are an expert marine-biology assistant for seaweed health assessment. "
        "You must strictly follow the model result provided by the user. "
        "Do not change the predicted class or confidence. "
        "Start by explicitly restating the exact model prediction and confidence. "
        "If the confidence is modest or the probabilities are close, explain that the model is uncertain. "
        "If the visual evidence might appear mixed, explain that the model prediction and human visual impression may not fully align. "
        "Write only 3 to 5 sentences."
    )

    prompt_payload = {
        "task": "Explain the seaweed prediction while strictly matching the exact model output.",
        "best_model": model_name,
        "prediction": {
            "predicted_class": pred_class,
            "confidence_percent": round(confidence_pct, 2),
            "healthy_probability_percent": round(healthy_pct, 2),
            "unhealthy_probability_percent": round(unhealthy_pct, 2),
        },
        "uncertainty_flag": is_uncertain,
        "image_summary": image_summary,
        "gradcam_summary": gradcam_summary,
        "true_label_if_known": true_label,
        "instruction": (
            f"The model prediction is EXACTLY '{pred_class}' at {confidence_pct:.2f}% confidence. "
            "Your first sentence must repeat that exact result. "
            "If the probabilities are close, mention that the classification is borderline or uncertain. "
            "If the image may visually seem to suggest the opposite class, acknowledge that possibility and say the model may be responding to subtle features or limited confidence. "
            "Do not invent a different predicted class."
        ),
    }

    explanation = call_deepseek_api(
        prompt=json.dumps(prompt_payload, indent=2),
        system_message=system_message,
        temperature=0.1,
        max_tokens=220,
    )

    lower_exp = explanation.lower()
    if pred_class == "healthy" and "predicted 'unhealthy'" in lower_exp:
        explanation = (
            f"The model predicted 'healthy' with {confidence_pct:.2f}% confidence. "
            f"The generated explanation below appeared inconsistent, so this corrected statement is being applied.\n\n{explanation}"
        )
    elif pred_class == "unhealthy" and "predicted 'healthy'" in lower_exp:
        explanation = (
            f"The model predicted 'unhealthy' with {confidence_pct:.2f}% confidence. "
            f"The generated explanation below appeared inconsistent, so this corrected statement is being applied.\n\n{explanation}"
        )

    return explanation


def analyze_for_api(image_path: str) -> Dict[str, Any]:
    result = analyze_seaweed_with_best_model(
        image_path=image_path,
        true_label=None,
        alpha=0.4,
        gradcam_threshold=0.6,
    )

    explanation = generate_deepseek_explanation_from_result(result)

    return {
        "model_name": "ConvNeXt-Tiny",
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "gradcam_summary": result["gradcam_summary"],
        "image_summary": result["image_summary"],
        "explanation": explanation,
        "original_image_base64": pil_to_base64_png(result["original_pil"]),
        "cropped_image_base64": pil_to_base64_png(result["cropped_pil"]),
        "overlay_image_base64": numpy_rgb_to_base64_png(result["overlay_np"]),
        "heatmap_image_base64": numpy_rgb_to_base64_png(result["heatmap_np"]),
    }