"""Embedded-text extraction (Auto mode), page order and blank pages."""

import pytest

from conftest import digital_pdf
from ocr_app import pipeline
from ocr_app.pipeline import METHOD_EMBEDDED, MODE_AUTO, process_pdf


@pytest.fixture
def ocr_must_not_run(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("OCR must not run for pages with embedded text")

    monkeypatch.setattr(pipeline.TesseractOCR, "recognize", fail)
    monkeypatch.setattr(pipeline, "get_handwriting_recognizer", fail)


def test_extracts_embedded_text_without_ocr(ocr_must_not_run):
    text = "The quick brown fox jumps over the lazy dog. Invoice number 4711."
    result = process_pdf(digital_pdf([text]), MODE_AUTO)
    assert result.mode == MODE_AUTO
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.number == 1
    assert page.method == METHOD_EMBEDDED
    assert " ".join(page.text.split()) == text
    assert page.warnings == []


def test_multipage_order_and_page_numbers(ocr_must_not_run):
    texts = [f"This is page number {i} with unique token TOKEN{i:03d}" for i in range(1, 8)]
    progress = []
    result = process_pdf(digital_pdf(texts), MODE_AUTO, progress_callback=lambda d, t: progress.append((d, t)))
    assert [p.number for p in result.pages] == list(range(1, 8))
    for i, page in enumerate(result.pages, 1):
        assert f"TOKEN{i:03d}" in page.text
        assert all(f"TOKEN{j:03d}" not in page.text for j in range(1, 8) if j != i)
    assert progress == [(i, 7) for i in range(0, 8)]


def test_short_embedded_text_without_images_is_used(ocr_must_not_run):
    result = process_pdf(digital_pdf(["Page 3"]), MODE_AUTO)
    assert result.pages[0].text == "Page 3"
    assert result.pages[0].method == METHOD_EMBEDDED


def test_preview_text_contains_page_markers_in_order(ocr_must_not_run):
    result = process_pdf(digital_pdf(["First page text here, long enough.", "Second page text here, long enough."]))
    preview = result.to_text()
    first, second = preview.index("===== PAGE 1"), preview.index("===== PAGE 2")
    assert first < preview.index("First page") < second < preview.index("Second page")


def test_blank_page_in_auto_mode_yields_warning_not_failure(fake_tesseract):
    fake_tesseract.responder = lambda image, i: "  \n\x0c"  # Tesseract's output for a blank page
    pdf = digital_pdf(["A real page of digital text, long enough to be used.", ""])
    result = process_pdf(pdf, MODE_AUTO, tesseract_cmd=fake_tesseract.cmd)
    assert result.pages[0].text.startswith("A real page")
    blank = result.pages[1]
    assert blank.text == ""
    assert blank.warnings == ["No text recognized (blank page?)."]
    assert result.warnings == ["Page 2: No text recognized (blank page?)."]
    assert len(fake_tesseract.calls) == 1  # only the blank page needed OCR


def test_status_messages_are_reported(ocr_must_not_run):
    messages = []
    process_pdf(digital_pdf(["Enough embedded text on this page."] * 2), status_callback=messages.append)
    assert messages[0] == "Opening PDF…"
    assert any("Page 2 of 2" in m for m in messages)


def test_cancellation_stops_processing(ocr_must_not_run):
    calls = {"n": 0}

    def cancel_after_first_page():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(pipeline.ProcessingCancelled):
        process_pdf(digital_pdf(["page one text long enough"] * 3), cancel_check=cancel_after_first_page)
