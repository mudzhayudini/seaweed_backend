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

def summarize_visual_features(pil_img):
    """
    Simple visual summary derived from the uploaded image.
    This gives DeepSeek more evidence about the image itself.
    """
    img = np.array(pil_img.convert("RGB")).astype(np.float32)

    # Basic brightness
    brightness = float(img.mean() / 255.0)

    # Channel means
    r_mean = float(img[:, :, 0].mean() / 255.0)
    g_mean = float(img[:, :, 1].mean() / 255.0)
    b_mean = float(img[:, :, 2].mean() / 255.0)

    # Rough color interpretation
    if g_mean > r_mean and g_mean > b_mean:
        dominant_color = "green-dominant"
    elif r_mean > g_mean and r_mean > b_mean:
        dominant_color = "red/brown-dominant"
    elif b_mean > r_mean and b_mean > g_mean:
        dominant_color = "blue-dominant"
    else:
        dominant_color = "mixed"

    # Contrast / variation
    contrast = float(img.std() / 255.0)

    # Texture roughness proxy
    gray = img.mean(axis=2)
    texture_variation = float(gray.std() / 255.0)

    # Coarse interpretation
    if brightness < 0.25:
        brightness_desc = "dark"
    elif brightness < 0.55:
        brightness_desc = "moderately bright"
    else:
        brightness_desc = "bright"

    if contrast < 0.12:
        contrast_desc = "low contrast"
    elif contrast < 0.22:
        contrast_desc = "moderate contrast"
    else:
        contrast_desc = "high contrast"

    if texture_variation < 0.10:
        texture_desc = "visually smooth"
    elif texture_variation < 0.20:
        texture_desc = "moderately varied"
    else:
        texture_desc = "highly varied or irregular"

    return {
        "brightness": round(brightness, 4),
        "brightness_description": brightness_desc,
        "dominant_color": dominant_color,
        "channel_means": {
            "red": round(r_mean, 4),
            "green": round(g_mean, 4),
            "blue": round(b_mean, 4),
        },
        "contrast": round(contrast, 4),
        "contrast_description": contrast_desc,
        "texture_variation": round(texture_variation, 4),
        "texture_description": texture_desc,
    }

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
    visual_summary = summarize_visual_features(pil_img)
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
        "visual_summary": visual_summary,
        "overlay_np": overlay,
        "heatmap_np": heatmap,
        "cam_np": cam_resized,
        "true_label": true_label,
    }


def generate_deepseek_explanation_from_result(result):
    """
    Uses the exact already-computed model result.
    This guarantees the explanation matches the displayed prediction,
    while also using Grad-CAM and image-derived visual cues.
    """
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    gradcam_summary = result["gradcam_summary"]
    image_summary = result["image_summary"]
    visual_summary = result.get("visual_summary", {})
    true_label = result.get("true_label", None)
    model_name = result["model_name"]

    confidence_pct = confidence * 100
    healthy_pct = probs["healthy"] * 100
    unhealthy_pct = probs["unhealthy"] * 100

    # Borderline case if probabilities are close
    is_uncertain = abs(probs["healthy"] - probs["unhealthy"]) < 0.15

    system_message = (
        "You are an expert marine-biology assistant for seaweed health assessment. "
        "You must strictly follow the exact model prediction and confidence provided. "
        "Do not change the predicted class. "
        "Your explanation must do four things: "
        "(1) first restate the exact classification and confidence from the model, "
        "(2) justify that prediction using the Grad-CAM attention summary and the image-derived visual summary, "
        "(3) explain in short scientific language why the seaweed may be healthy or unhealthy, "
        "(4) if the visual evidence appears mixed or seems to suggest the opposite class, explicitly mention that the visual impression may partially contradict the model and explain that the model may be relying on subtle features or a borderline decision. "
        "Write 4 to 6 sentences only. "
        "Be cautious, specific, and consistent with the provided evidence."
    )

    prompt_payload = {
        "task": "Explain and justify the seaweed classification based on model output, Grad-CAM attention, and image-derived visual cues.",
        "best_model": model_name,
        "prediction": {
            "predicted_class": pred_class,
            "confidence_percent": round(confidence_pct, 2),
            "healthy_probability_percent": round(healthy_pct, 2),
            "unhealthy_probability_percent": round(unhealthy_pct, 2),
        },
        "uncertainty_flag": is_uncertain,
        "image_summary": image_summary,
        "visual_summary": visual_summary,
        "gradcam_summary": gradcam_summary,
        "true_label_if_known": true_label,
        "instruction": (
            f"The model prediction is EXACTLY '{pred_class}' at {confidence_pct:.2f}% confidence. "
            "Your first sentence must repeat that exact result. "
            "Then explain why this result is plausible using the visual summary and the Grad-CAM attention region. "
            "If the attention region supports the model, say so clearly. "
            "If the image-derived visual cues seem mixed, borderline, or partially opposite to the model class, explicitly mention that possible contradiction and explain that the model may be responding to subtle or localized features. "
            "Do not invent a different class prediction. "
            "Do not say the model predicted the opposite class."
        ),
    }

    explanation = call_deepseek_api(
        prompt=json.dumps(prompt_payload, indent=2),
        system_message=system_message,
        temperature=0.1,
        max_tokens=260
    )

    # Safety correction if the LLM still contradicts the model
    lower_exp = explanation.lower()

    if pred_class == "healthy" and "predicted 'unhealthy'" in lower_exp:
        explanation = (
            f"The model predicted 'healthy' with {confidence_pct:.2f}% confidence. "
            f"The explanation below contained an inconsistency, so this correction is applied first.\n\n{explanation}"
        )
    elif pred_class == "unhealthy" and "predicted 'healthy'" in lower_exp:
        explanation = (
            f"The model predicted 'unhealthy' with {confidence_pct:.2f}% confidence. "
            f"The explanation below contained an inconsistency, so this correction is applied first.\n\n{explanation}"
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

