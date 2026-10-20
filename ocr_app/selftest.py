"""Headless self-test, used to verify a packaged build: ``PDFtoWordOCR.exe --self-test report.json``.

Required checks: library imports, digital-PDF extraction, DOCX export and GUI
construction. Optional checks (reported as skipped when unavailable):
Tesseract OCR and the handwriting model from the local cache (never downloads).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, Optional


class _Skip(Exception):
    pass


def _check_imports() -> str:
    import cv2
    import docx  # noqa: F401
    import numpy
    import PIL
    import pymupdf
    import pytesseract  # noqa: F401
    import torch
    import transformers
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: F401

    return (
        f"pymupdf {pymupdf.__version__}, Pillow {PIL.__version__}, OpenCV {cv2.__version__}, "
        f"numpy {numpy.__version__}, torch {torch.__version__} (CUDA available: "
        f"{torch.cuda.is_available()}), transformers {transformers.__version__}"
    )


def _make_pdf(text: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def _check_digital_pdf() -> str:
    from .pipeline import METHOD_EMBEDDED, process_pdf

    result = process_pdf(_make_pdf("Self-test digital text extraction 12345"), "Auto")
    page = result.pages[0]
    if "12345" not in page.text or page.method != METHOD_EMBEDDED:
        raise AssertionError(f"unexpected result: {page}")
    return f"{page.method}: {page.text!r}"


def _check_image_input() -> str:
    import io

    from PIL import Image

    from .pipeline import open_pdf

    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, format="PNG")
    doc = open_pdf(buf.getvalue())
    pages = doc.page_count
    doc.close()
    if pages != 1:
        raise AssertionError(f"expected 1 page, got {pages}")
    return "PNG image opened as a 1-page document"


def _check_docx(tmp: Path) -> str:
    from docx import Document

    from .docx_export import export_docx

    path = export_docx("===== PAGE 1 · x =====\nHello Word\n\n===== PAGE 2 · x =====\nSecond page", tmp / "t.docx")
    texts = [p.text for p in Document(str(path)).paragraphs if p.text]
    if texts != ["Hello Word", "Second page"]:
        raise AssertionError(f"unexpected paragraphs: {texts}")
    return f"wrote and re-read {path.name}"


def _check_gui(window_factory: Optional[Callable]) -> str:
    if window_factory is None:
        raise _Skip("no window factory supplied")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    settings = QSettings(str(Path(tempfile.gettempdir()) / "pdftowordocr_selftest.ini"), QSettings.Format.IniFormat)
    window = window_factory(settings)
    title = window.windowTitle()
    window.close()
    window.deleteLater()
    app.processEvents()
    return f"main window created ({title}, platform {app.platformName()})"


def _check_tesseract(tmp: Path) -> str:
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont

    from .pipeline import METHOD_TESSERACT, TesseractNotFoundError, find_tesseract, process_pdf, tesseract_version

    try:
        version = tesseract_version()
    except TesseractNotFoundError as exc:
        raise _Skip(str(exc).splitlines()[0])
    cmd = find_tesseract()
    image = Image.new("RGB", (1400, 300), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default(size=64)
    ImageDraw.Draw(image).text((40, 100), "QUICK BROWN FOX 2024", fill="black", font=font)
    image_path = tmp / "scan.png"
    image.save(image_path)
    doc = pymupdf.open()
    page = doc.new_page(width=700, height=150)
    page.insert_image(page.rect, filename=str(image_path))
    data = doc.tobytes()
    doc.close()
    result = process_pdf(data, "Printed Text", dpi=200)
    text = result.pages[0].text
    if result.pages[0].method != METHOD_TESSERACT or "BROWN" not in text.upper():
        raise AssertionError(f"unexpected OCR output: {text!r}")
    return f"Tesseract {version} ({cmd}): {text!r}"


def _check_handwriting() -> str:
    import cv2
    import numpy as np
    from PIL import Image

    from .pipeline import HandwritingModelError, HandwritingRecognizer, resolve_model_source

    recognizer = HandwritingRecognizer(allow_download=False)
    source = resolve_model_source(recognizer.model_id)
    try:
        recognizer.load()
    except HandwritingModelError as exc:
        raise _Skip(str(exc).splitlines()[0])
    canvas = np.full((220, 1100, 3), 255, np.uint8)
    cv2.putText(canvas, "hello world", (40, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (0, 0, 0), 4, cv2.LINE_AA)
    start = time.perf_counter()
    text = recognizer.recognize_page(Image.fromarray(canvas))
    elapsed = time.perf_counter() - start
    if not text.strip():
        raise AssertionError("model returned no text")
    return f"model {source}, device {recognizer.device}, {elapsed:.1f}s: {text!r}"


def run_self_test(report_path: Optional[str] = None, window_factory: Optional[Callable] = None) -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix="pdfocr_selftest_") as tmp_name:
        tmp = Path(tmp_name)
        checks = [
            ("imports", True, _check_imports),
            ("digital_pdf", True, _check_digital_pdf),
            ("image_input", True, _check_image_input),
            ("docx_export", True, lambda: _check_docx(tmp)),
            ("gui", True, lambda: _check_gui(window_factory)),
            ("tesseract_ocr", False, lambda: _check_tesseract(tmp)),
            ("handwriting_model", False, _check_handwriting),
        ]
        for name, required, check in checks:
            try:
                detail, status = check(), "pass"
            except _Skip as exc:
                detail, status = str(exc), "skipped"
            except Exception as exc:
                detail, status = f"{exc}\n{traceback.format_exc()}", "fail"
            results.append({"check": name, "required": required, "status": status, "detail": detail})

    ok = all(r["status"] == "pass" for r in results if r["required"]) and not any(
        r["status"] == "fail" for r in results
    )
    report = {"ok": ok, "frozen": bool(getattr(sys, "frozen", False)), "python": sys.version, "checks": results}
    text = json.dumps(report, indent=2)
    if report_path:
        Path(report_path).write_text(text, encoding="utf-8")
    if sys.stdout is not None:
        print(text)
    return 0 if ok else 1
