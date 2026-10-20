"""Shared fixtures: generated PDFs, a fake Tesseract and a fake TrOCR model.

No test here needs internet access, model weights, a GPU or a real Tesseract
install; the few integration tests that use real tools skip themselves when
those are unavailable.
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
import pytest  # noqa: E402
import pytesseract  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from ocr_app import pipeline  # noqa: E402


# --------------------------------------------------------------------------- PDF builders


def digital_pdf(page_texts: list[str]) -> bytes:
    """A PDF whose pages contain real (embedded) text; '' gives a blank page."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def text_image(text: str, size=(1200, 300)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text((40, size[1] // 3), text, fill="black", font_size=48)
    return image


def handwriting_image(lines: list[str], size=(1700, 1400)) -> Image.Image:
    """White page with script-font lines (a stand-in for handwriting)."""
    canvas = np.full((size[1], size[0], 3), 255, np.uint8)
    for i, line in enumerate(lines):
        y = 220 + i * 320
        cv2.putText(canvas, line, (80, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.2, (20, 20, 20), 4, cv2.LINE_AA)
    return Image.fromarray(canvas)


def image_pdf(images: list[Image.Image | None], page_size=(612, 792)) -> bytes:
    """A scanned-style PDF: each page holds one full-page image (None = blank page)."""
    doc = pymupdf.open()
    for image in images:
        page = doc.new_page(width=page_size[0], height=page_size[1])
        if image is not None:
            page.insert_image(page.rect, stream=png_bytes(image))
    data = doc.tobytes()
    doc.close()
    return data


# --------------------------------------------------------------------------- Tesseract


TSV_HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


class FakeTesseract:
    def __init__(self, exe_path: str):
        self.cmd = exe_path
        self.calls: list[dict] = []
        self.languages = ["eng", "fra", "osd"]
        self.confidence = 95.0  # reported for every word
        self.responder = lambda image, call_index: f"Recognized printed text {call_index}"

    def _run(self, image, lang, api):
        self.calls.append({"size": image.size, "lang": lang, "api": api, "cmd": pytesseract.pytesseract.tesseract_cmd})
        return self.responder(image, len(self.calls))

    def image_to_string(self, image, lang=None, config=""):
        return self._run(image, lang, f"image_to_string {config}")

    def run_and_get_multiple_output(self, image, extensions, lang=None, **_kwargs):
        assert extensions == ["txt", "tsv"]
        text = self._run(image, lang, "txt+tsv")
        rows = [TSV_HEADER, "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t"]
        rows += [f"5\t1\t1\t1\t1\t{i}\t0\t0\t10\t10\t{self.confidence}\t{w}" for i, w in enumerate(text.split(), 1)]
        return [text, "\n".join(rows)]


@pytest.fixture
def fake_tesseract(monkeypatch, tmp_path):
    """Patch pytesseract so no real Tesseract is needed; exposes recorded calls."""
    exe = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    fake = FakeTesseract(str(exe))
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    monkeypatch.setattr(pytesseract, "image_to_string", fake.image_to_string)
    monkeypatch.setattr(pytesseract, "run_and_get_multiple_output", fake.run_and_get_multiple_output)
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": list(fake.languages))
    return fake


@pytest.fixture
def no_tesseract(monkeypatch, tmp_path):
    """Make auto-detection find nothing, whatever is installed on this machine."""
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    monkeypatch.setattr(pipeline, "_default_tesseract_locations", lambda: [tmp_path / "missing" / "tesseract.exe"])
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")


# --------------------------------------------------------------------------- TrOCR


class FakeBatch:
    def __init__(self, pixel_values):
        self.pixel_values = pixel_values


class FakeProcessor:
    """Mimics TrOCRProcessor: records crops, decodes ids to 'line N'."""

    def __init__(self):
        self.calls: list[list[Image.Image]] = []

    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        self.calls.append(list(images))
        return FakeBatch(torch.zeros(len(images), 3, 8, 8))

    def batch_decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return [f" line {int(row[0])} " for row in ids]

    @property
    def crops(self) -> list[Image.Image]:
        return [crop for call in self.calls for crop in call]


class FakeModel:
    """Mimics VisionEncoderDecoderModel.generate: returns sequential ids."""

    def __init__(self):
        self.counter = 0
        self.devices: list[str] = []
        self.generate_devices: list[str] = []
        self.generate_options: list[dict] = []

    def to(self, device):
        self.devices.append(str(device))
        return self

    def eval(self):
        return self

    def generate(self, pixel_values, max_new_tokens, **options):
        assert max_new_tokens > 0
        self.generate_devices.append(str(pixel_values.device))
        self.generate_options.append(options)
        n = pixel_values.shape[0]
        ids = torch.arange(self.counter + 1, self.counter + n + 1).unsqueeze(1)
        self.counter += n
        return ids


@pytest.fixture
def fake_recognizer():
    return pipeline.HandwritingRecognizer(processor=FakeProcessor(), model=FakeModel(), device="cpu", batch_size=2)


@pytest.fixture(autouse=True)
def _isolate_recognizer_cache(monkeypatch):
    monkeypatch.setattr(pipeline, "_recognizers", {})


@pytest.fixture(autouse=True)
def _isolate_app_folder(monkeypatch, tmp_path_factory):
    """Hide any real models\\ or tesseract\\ folder beside the app from the tests."""
    empty = tmp_path_factory.mktemp("app_dir")
    monkeypatch.setattr(pipeline, "app_base_dir", lambda: empty)
