"""Image clean-up helpers used before OCR of photos and scans."""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr_app import imaging


def page_with_text(size=(1200, 900), colour=(0, 0, 0), paper=(250, 250, 250)) -> np.ndarray:
    image = Image.new("RGB", size, paper)
    draw = ImageDraw.Draw(image)
    for i in range(8):
        draw.text((80, 80 + i * 90), f"Line {i} of receipt text TOTAL 12.50", fill=colour, font_size=40)
    return np.asarray(image)


def test_best_gray_keeps_purple_ink_visible():
    # Faint purple print: in luminance it is much paler than in the green channel.
    rgb = page_with_text(colour=(190, 120, 205), paper=(235, 235, 235))
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chosen = imaging.best_gray(rgb)
    def contrast(c):
        return np.percentile(c, 50) - np.percentile(c, 1)
    assert contrast(chosen) > contrast(luma) * 1.15
    assert np.array_equal(chosen, rgb[:, :, 1])


def test_best_gray_uses_luminance_for_black_ink():
    rgb = page_with_text()
    assert np.array_equal(imaging.best_gray(rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def test_paper_box_finds_a_receipt_on_a_dark_table():
    photo = np.full((1600, 1200), 60, np.uint8)  # dark table
    photo[300:1400, 400:800] = 235  # receipt
    x0, y0, x1, y1 = imaging.paper_box(photo)
    assert abs(x0 - 400) < 30 and abs(x1 - 800) < 30 and abs(y0 - 300) < 30 and abs(y1 - 1400) < 30


def test_paper_box_leaves_a_plain_scan_alone():
    gray = cv2.cvtColor(page_with_text(), cv2.COLOR_RGB2GRAY)
    assert imaging.paper_box(gray) is None


def test_flatten_evens_out_a_shadow():
    gray = cv2.cvtColor(page_with_text(), cv2.COLOR_RGB2GRAY).astype(np.float32)
    shadow = np.linspace(1.0, 0.45, gray.shape[1])[None, :]  # darker to the right
    shaded = (gray * shadow).astype(np.uint8)
    assert imaging.lighting_is_uneven(shaded)
    flat = imaging.flatten(shaded)
    paper_left, paper_right = np.median(flat[:, 20:120]), np.median(flat[:, -120:-20])
    assert abs(float(paper_left) - float(paper_right)) < 15
    assert not imaging.lighting_is_uneven(flat)


def test_otsu_ink_fraction_flags_a_dark_backdrop():
    gray = cv2.cvtColor(page_with_text(), cv2.COLOR_RGB2GRAY)
    assert imaging.otsu_ink_fraction(gray) < 0.15
    backdrop = np.full((1600, 1200), 40, np.uint8)
    backdrop[400:1200, 300:900] = 240
    assert imaging.otsu_ink_fraction(backdrop) > 0.2


@pytest.mark.parametrize("angle", [-6.0, 3.0])
def test_skew_angle_recovers_rotation(angle):
    gray = cv2.cvtColor(page_with_text(), cv2.COLOR_RGB2GRAY)
    tilted = imaging.rotate(gray, -angle)  # tilt the page by -angle ...
    found = imaging.skew_angle(imaging.binarize(tilted))
    assert found == pytest.approx(angle, abs=0.6)  # ... and the correction is +angle


def test_skew_angle_of_a_blank_page_is_zero():
    assert imaging.skew_angle(np.zeros((500, 500), np.uint8)) == 0.0


def test_rotate_expands_and_fills_white():
    gray = np.zeros((100, 200), np.uint8)
    rotated = imaging.rotate(gray, 30)
    assert rotated.shape[0] > 100 and rotated.shape[1] > 180
    assert rotated[0, 0] == 255  # new corner area is white


def test_remove_rules_deletes_lines_but_keeps_writing():
    ink = np.zeros((400, 800), np.uint8)
    ink[200, 20:780] = 255  # long ruled line
    ink[20:380, 700] = 255  # long vertical box edge
    cv2.putText(ink, "hello", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4)
    cleaned = imaging.remove_rules(ink)
    assert cleaned[200, 400] == 0 and cleaned[300, 700] == 0
    assert cleaned[100:160, 90:330].sum() > 0.8 * ink[100:160, 90:330].sum()


def test_prepare_photo_crops_flattens_and_straightens():
    page = Image.fromarray(page_with_text(size=(800, 600)))
    canvas = Image.new("RGB", (1400, 1200), (50, 45, 40))  # dark table
    canvas.paste(page.rotate(5, expand=True, fillcolor=(250, 250, 250)), (250, 250))
    cleaned, done = imaging.prepare_photo(canvas)
    assert "crop" in done
    assert done.get("deskew") == pytest.approx(-5, abs=0.8)
    assert cleaned.shape[0] < 1200 and cleaned.shape[1] < 1400


def test_prepare_photo_leaves_a_clean_scan_unchanged():
    page = Image.fromarray(page_with_text())
    cleaned, done = imaging.prepare_photo(page)
    assert done == {}
    assert np.array_equal(cleaned, np.asarray(page.convert("L")))
