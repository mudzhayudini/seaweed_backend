from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def segment_and_crop_seaweed(pil_img: Image.Image, padding: int = 10):
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    rect = (
        max(1, int(0.05 * w)),
        max(1, int(0.05 * h)),
        max(2, int(0.90 * w)),
        max(2, int(0.90 * h)),
    )

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype("uint8")

    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return pil_img, (fg_mask * 255), img

    x_min = max(0, xs.min() - padding)
    x_max = min(w, xs.max() + padding)
    y_min = max(0, ys.min() - padding)
    y_max = min(h, ys.max() + padding)

    segmented = img.copy()
    segmented[fg_mask == 0] = 0

    cropped = segmented[y_min:y_max, x_min:x_max]
    cropped_pil = Image.fromarray(cropped)
    mask_uint8 = (fg_mask * 255).astype(np.uint8)

    return cropped_pil, mask_uint8, segmented