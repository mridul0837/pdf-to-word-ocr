"""Image clean-up for OCR of photos and scans (pure functions on numpy greyscale arrays).

Phone photos and phone-scanner PDFs break the assumptions of a single global threshold:
light falls unevenly, paper is grey or coloured, the document lies on a darker table,
thermal or ballpoint ink is faint or coloured, and pages are tilted. These helpers fix
those conditions; the OCR engines decide whether to use them by comparing confidence.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def best_gray(rgb: np.ndarray) -> np.ndarray:
    """Greyscale that keeps coloured ink visible.

    Luminance can wash out purple thermal print or blue ink. Pick, among luminance and the
    three colour channels, the one where the darkest pixels (ink) stand furthest from the
    typical (paper) brightness.
    """
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    candidates = [luma, rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]

    def contrast(channel: np.ndarray) -> float:
        sample = channel[::4, ::4]
        return float(np.percentile(sample, 50) - np.percentile(sample, 1))

    best = max(candidates, key=contrast)
    # Prefer luminance unless a channel is clearly better (avoids needless colour switching).
    return luma if contrast(best) < contrast(luma) * 1.15 else np.ascontiguousarray(best)


def paper_box(gray: np.ndarray) -> Optional[Box]:
    """Bounding box of the document when it lies on a clearly darker background.

    Returns None when no such region stands out (scans, full-frame photos), so callers can
    keep the whole image.
    """
    region = paper_region(gray)
    return region[0] if region else None


def paper_region(gray: np.ndarray) -> Optional[tuple[Box, np.ndarray]]:
    """Bounding box and full-size mask (255 = paper) of a document on a darker background."""
    height, width = gray.shape
    scale = 0.25 if max(height, width) > 1200 else 0.5
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (21, 21), 0)
    threshold, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright, dark = blur[mask > 0], blur[mask == 0]
    if bright.size == 0 or dark.size == 0 or float(bright.mean()) - float(dark.mean()) < 40:
        return None  # no distinct darker background
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if not 0.12 < (w * h) / (small.shape[0] * small.shape[1]) < 0.92:
        return None
    pad = 3
    box = (max(0, int((x - pad) / scale)), max(0, int((y - pad) / scale)),
           min(width, int((x + w + pad) / scale)), min(height, int((y + h + pad) / scale)))
    region = np.zeros_like(small)
    cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
    return box, cv2.resize(region, (width, height), interpolation=cv2.INTER_NEAREST)


def background(gray: np.ndarray) -> np.ndarray:
    """Estimate the paper brightness everywhere by removing dark strokes (max filter)."""
    k = max(15, (min(gray.shape) // 40) | 1)
    estimate = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.medianBlur(estimate, 21)


def flatten(gray: np.ndarray) -> np.ndarray:
    """Even out lighting: divide by the estimated paper brightness, then stretch contrast."""
    flat = cv2.divide(gray, background(gray), scale=255)
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)


def lighting_is_uneven(gray: np.ndarray) -> bool:
    """True when the paper brightness varies a lot across the page (shadows, gradients)."""
    sample = background(cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    return float(np.percentile(sample, 95) - np.percentile(sample, 5)) > 35


def otsu_ink_fraction(gray: np.ndarray) -> float:
    """Share of pixels a global Otsu threshold calls ink; text pages are usually < 15%."""
    _, binary = cv2.threshold(cv2.medianBlur(gray, 3), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((binary > 0).mean())


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask (255 = ink) with a local threshold that tolerates uneven lighting."""
    block = max(31, (min(gray.shape) // 30) | 1)
    return cv2.adaptiveThreshold(cv2.medianBlur(gray, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 15)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Ink (255) for finding handwriting: locally dark *and* clearly darker than the paper.

    A local threshold alone also picks up faint printed rules and paper texture; a global
    one loses writing under uneven light. The contrast gate scales with how dark this
    page's writing is (40% of the typical ink depth, at least 25 levels), so faint ruled
    lines on a page of bold pen are ignored while faint pencil on its own page is kept.
    """
    local = binarize(gray)
    depth = cv2.subtract(background(gray), gray)  # how much darker than the paper here
    strong = float(np.percentile(depth[local > 0], 90)) if np.count_nonzero(local) else 0.0
    gate = max(25.0, 0.4 * strong)
    return cv2.bitwise_and(local, np.where(depth > gate, 255, 0).astype(np.uint8))


def skew_angle(ink: np.ndarray, limit: float = 15.0) -> float:
    """Dominant text-line angle in degrees (positive = counter-clockwise), found by the
    rotation that makes the row profile of the ink sharpest. 0.0 if there is little ink."""
    if ink.mean() < 0.5:  # mean of 0/255 values: < 0.2% ink
        return 0.0
    scale = min(1.0, 800 / max(ink.shape))
    small = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    centre = (small.shape[1] / 2, small.shape[0] / 2)

    def sharpness(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(small, matrix, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse = max(np.arange(-limit, limit + 0.01, 1.0), key=sharpness)
    fine = max(np.arange(coarse - 1.0, coarse + 1.01, 0.1), key=sharpness)
    return float(round(fine, 1))


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the centre, expanding the canvas; new areas are filled with white."""
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(height * sin + width * cos), int(height * cos + width * sin)
    matrix[0, 2] += new_w / 2 - width / 2
    matrix[1, 2] += new_h / 2 - height / 2
    return cv2.warpAffine(gray, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=255)


def remove_specks(ink: np.ndarray) -> np.ndarray:
    """Drop connected blobs far too small to be writing (paper texture, dust, noise)."""
    min_area = max(6, int((min(ink.shape) / 300) ** 2 * 2))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def remove_frames(ink: np.ndarray) -> np.ndarray:
    """Drop page edges, card outlines and form frames: shapes that are very large (over half
    the width or a third of the height) yet almost empty (under 10% of their bounding box).
    Unlike ``remove_rules`` this also catches tilted, curved or broken borders. Writing
    never has that shape."""
    height, width = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    frame = ((w > 0.5 * width) | (h > 0.33 * height)) & (area < 0.1 * w * h)
    # Streaks along the photo's edge (paper edge, shadow): tall, thin and touching a border.
    # Writing cut off by the edge is word-shaped, not like this.
    touches = (x <= 1) | (y <= 1) | (x + w >= width - 1) | (y + h >= height - 1)
    frame |= touches & (h > 3 * w) & (h > 0.05 * height)
    frame[0] = False  # background label
    return np.where(frame[labels], 0, ink).astype(np.uint8)


def remove_rules(ink: np.ndarray) -> np.ndarray:
    """Remove long thin horizontal and vertical lines (ruled paper, form boxes) from an ink mask.

    Only straight runs much longer than any letter stroke count as rules (an eighth of the
    page width, a sixth of its height), so large handwriting is not cut up.
    """
    height, width = ink.shape
    lines = np.zeros_like(ink)
    for size in ((max(80, width // 8), 1), (1, max(80, height // 6))):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(lines))


def prepare_photo(image: Image.Image) -> tuple[np.ndarray, dict]:
    """Clean-up for a photographed page: crop to the paper, choose a greyscale that keeps
    coloured ink, even out the lighting when needed and straighten the text.

    Returns the cleaned greyscale and a record of what was done.
    """
    gray = best_gray(to_array(image))
    done: dict = {}
    region = paper_region(gray)
    if region is not None:
        (x0, y0, x1, y1), mask = region
        # Paint everything outside the paper (table corners around a tilted page) paper-white,
        # so it cannot pass for text or skew the angle estimate.
        paper_level = np.uint8(np.percentile(gray[mask > 0], 90))
        gray = np.where(mask > 0, gray, paper_level).astype(np.uint8)[y0:y1, x0:x1]
        done["crop"] = (x0, y0, x1, y1)
    if lighting_is_uneven(gray) or otsu_ink_fraction(gray) > 0.2:
        gray = flatten(gray)
        done["flatten"] = True
    angle = skew_angle(binarize(gray))
    if abs(angle) >= 0.5:
        gray = rotate(gray, angle)
        done["deskew"] = angle
    return gray, done
