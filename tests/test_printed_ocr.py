"""Printed OCR via Tesseract (mocked), plus an optional real-Tesseract integration test."""

import pytesseract
import pytest

from conftest import digital_pdf, image_pdf, text_image
from ocr_app import pipeline
from ocr_app.pipeline import (
    METHOD_EMBEDDED,
    METHOD_FAILED,
    METHOD_TESSERACT,
    MODE_AUTO,
    MODE_PRINTED,
    TesseractLanguageError,
    TesseractNotFoundError,
    process_pdf,
)


def test_printed_mode_ocrs_rendered_pages(fake_tesseract):
    pdf = image_pdf([text_image("Scanned line")])
    result = process_pdf(pdf, MODE_PRINTED, language="eng", dpi=200, tesseract_cmd=fake_tesseract.cmd)
    assert [p.text for p in result.pages] == ["Recognized printed text 1"]
    assert result.pages[0].method == METHOD_TESSERACT
    call = fake_tesseract.calls[0]
    assert call["lang"] == "eng"
    assert call["api"] == "txt+tsv"  # one Tesseract run returns text and confidences
    assert call["cmd"] == fake_tesseract.cmd
    # US Letter (8.5 x 11 in) rendered at 200 DPI.
    assert call["size"] == (1700, 2200)


def test_printed_mode_ignores_embedded_text(fake_tesseract):
    pdf = digital_pdf(["Embedded text that Printed mode should not use directly."])
    result = process_pdf(pdf, "Printed Text", tesseract_cmd=fake_tesseract.cmd)
    assert result.pages[0].method == METHOD_TESSERACT
    assert len(fake_tesseract.calls) == 1


def test_printed_mode_keeps_page_order(fake_tesseract):
    # Each page has a different width, so the rendered image identifies the page.
    import pymupdf

    doc = pymupdf.open()
    for i in range(5):
        doc.new_page(width=300 + 100 * i, height=400).insert_text((20, 50), f"p{i}")
    pdf_pages = doc.tobytes()
    doc.close()
    fake_tesseract.responder = lambda image, i: f"width={image.size[0]}"
    result = process_pdf(pdf_pages, MODE_PRINTED, dpi=72, tesseract_cmd=fake_tesseract.cmd)
    assert [p.text for p in result.pages] == [f"width={300 + 100 * i}" for i in range(5)]
    assert [p.number for p in result.pages] == [1, 2, 3, 4, 5]


def test_auto_mode_mixes_embedded_text_and_ocr(fake_tesseract):
    import pymupdf

    digital = pymupdf.open(stream=digital_pdf(["Digital page one with plenty of embedded text.",
                                               "Digital page three with plenty of embedded text."]))
    scanned = pymupdf.open(stream=image_pdf([text_image("scan")]))
    digital.insert_pdf(scanned, start_at=1)
    data = digital.tobytes()
    fake_tesseract.responder = lambda image, i: "Text from the scanned page\n\x0c"
    result = process_pdf(data, MODE_AUTO, tesseract_cmd=fake_tesseract.cmd)
    assert [p.method for p in result.pages] == [METHOD_EMBEDDED, METHOD_TESSERACT, METHOD_EMBEDDED]
    assert result.pages[1].text == "Text from the scanned page"  # form feed removed
    assert len(fake_tesseract.calls) == 1


def test_empty_ocr_output_produces_blank_page_warning(fake_tesseract):
    fake_tesseract.responder = lambda image, i: ""
    result = process_pdf(image_pdf([None, text_image("x")]), MODE_PRINTED, tesseract_cmd=fake_tesseract.cmd)
    assert [p.text for p in result.pages] == ["", ""]
    assert all(p.warnings == ["No text recognized (blank page?)."] for p in result.pages)


def test_ocr_failure_on_one_page_is_reported_and_processing_continues(fake_tesseract):
    def responder(image, i):
        if i == 2:
            raise pytesseract.TesseractError(1, "Image too large")
        return f"page {i} ok"

    fake_tesseract.responder = responder
    pdf = image_pdf([text_image("a"), text_image("b"), text_image("c")])
    result = process_pdf(pdf, MODE_PRINTED, tesseract_cmd=fake_tesseract.cmd)
    assert [p.text for p in result.pages] == ["page 1 ok", "", "page 3 ok"]
    assert result.pages[1].method == METHOD_FAILED
    assert "Tesseract failed: Image too large" in result.pages[1].warnings[0]


@pytest.mark.parametrize("mode", [MODE_PRINTED, MODE_AUTO])
def test_low_confidence_page_suggests_handwriting_mode(fake_tesseract, mode):
    fake_tesseract.confidence = 47.0  # what Tesseract reported on a real handwritten letter
    fake_tesseract.responder = lambda image, i: "1 ZS Ock VA From: Venel TCO"
    result = process_pdf(image_pdf([text_image("x")]), mode, tesseract_cmd=fake_tesseract.cmd)
    page = result.pages[0]
    assert page.text == "1 ZS Ock VA From: Venel TCO"  # text is still returned
    assert len(page.warnings) == 1
    assert "Low OCR confidence (47%)" in page.warnings[0]
    assert "use Handwriting mode" in page.warnings[0]


def test_confident_page_has_no_hint(fake_tesseract):
    fake_tesseract.confidence = 96.0
    result = process_pdf(image_pdf([text_image("x")]), MODE_PRINTED, tesseract_cmd=fake_tesseract.cmd)
    assert result.pages[0].warnings == []


def test_mean_word_confidence_ignores_non_word_rows():
    from conftest import TSV_HEADER

    tsv = "\n".join([
        TSV_HEADER,
        "1\t1\t0\t0\t0\t0\t0\t0\t9\t9\t-1\t",        # page row
        "5\t1\t1\t1\t1\t1\t0\t0\t9\t9\t90.5\tHello",
        "5\t1\t1\t1\t1\t2\t0\t0\t9\t9\t40\tw0rld",
        "5\t1\t1\t1\t1\t3\t0\t0\t9\t9\t95\t ",        # empty word
        "5\t1\t1\t1\t1\t4\t0\t0\t9\t9\t-1\tx",        # no confidence
    ])
    assert pipeline.mean_word_confidence(tsv) == pytest.approx(65.25)
    assert pipeline.mean_word_confidence(TSV_HEADER) is None


def test_missing_language_data_is_a_clear_error(fake_tesseract):
    with pytest.raises(TesseractLanguageError, match="not installed: deu"):
        process_pdf(image_pdf([text_image("x")]), MODE_PRINTED, language="eng+deu", tesseract_cmd=fake_tesseract.cmd)


def test_tesseract_without_any_language_data_is_a_clear_error(fake_tesseract):
    fake_tesseract.languages = []
    with pytest.raises(TesseractLanguageError, match="TESSDATA_PREFIX"):
        process_pdf(image_pdf([text_image("x")]), MODE_PRINTED, tesseract_cmd=fake_tesseract.cmd)
    assert fake_tesseract.calls == []


def test_invalid_language_string_is_rejected(fake_tesseract):
    with pytest.raises(TesseractLanguageError, match="Invalid Tesseract language"):
        process_pdf(image_pdf([text_image("x")]), MODE_PRINTED, language="eng; rm -rf", tesseract_cmd=fake_tesseract.cmd)


# --------------------------------------------------------------------------- missing Tesseract


def test_configured_path_that_does_not_exist(tmp_path):
    with pytest.raises(TesseractNotFoundError, match="configured Tesseract path does not exist"):
        pipeline.find_tesseract(str(tmp_path / "nowhere" / "tesseract.exe"))


def test_configured_directory_resolves_to_executable(fake_tesseract):
    from pathlib import Path

    assert pipeline.find_tesseract(str(Path(fake_tesseract.cmd).parent)) == fake_tesseract.cmd
    assert pipeline.find_tesseract(f'"{fake_tesseract.cmd}"') == fake_tesseract.cmd


def test_environment_variable_is_used(monkeypatch, no_tesseract, fake_tesseract):
    monkeypatch.setenv("TESSERACT_CMD", fake_tesseract.cmd)
    assert pipeline.find_tesseract() == fake_tesseract.cmd


def test_tesseract_folder_next_to_the_app_is_found(monkeypatch, tmp_path):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(pipeline, "app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: r"C:\elsewhere\tesseract.exe")
    exe = tmp_path / "tesseract" / pipeline.TESSERACT_EXE
    exe.parent.mkdir()
    exe.write_bytes(b"")
    assert pipeline.find_tesseract() == str(exe)  # bundled copy wins over PATH


def test_auto_detection_failure_explains_how_to_fix(no_tesseract):
    with pytest.raises(TesseractNotFoundError, match="UB-Mannheim"):
        pipeline.find_tesseract()


def test_printed_mode_without_tesseract_raises(no_tesseract):
    with pytest.raises(TesseractNotFoundError, match="Tesseract OCR was not found"):
        process_pdf(image_pdf([text_image("x")]), MODE_PRINTED)


def test_auto_mode_without_tesseract_keeps_digital_pages(no_tesseract):
    import pymupdf

    doc = pymupdf.open(stream=digital_pdf(["Digital page with a sufficient amount of text."]))
    doc.insert_pdf(pymupdf.open(stream=image_pdf([text_image("scan")])))
    result = process_pdf(doc.tobytes(), MODE_AUTO)
    assert result.pages[0].text.startswith("Digital page")
    assert result.pages[1].text == ""
    assert "Tesseract is unavailable" in result.pages[1].warnings[0]


def test_auto_mode_without_tesseract_and_no_text_raises(no_tesseract):
    with pytest.raises(TesseractNotFoundError):
        process_pdf(image_pdf([text_image("scan"), text_image("scan 2")]), MODE_AUTO)


def test_tesseract_that_cannot_run_is_reported(monkeypatch, fake_tesseract):
    def broken(config=""):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "get_languages", broken)
    with pytest.raises(TesseractNotFoundError, match="could not be run"):
        process_pdf(image_pdf([text_image("x")]), MODE_PRINTED, tesseract_cmd=fake_tesseract.cmd)


def test_tesseract_vanishing_mid_run_is_reported(monkeypatch, fake_tesseract):
    def vanish(image, i):
        raise pytesseract.TesseractNotFoundError()

    fake_tesseract.responder = vanish
    with pytest.raises(TesseractNotFoundError):
        process_pdf(image_pdf([text_image("x")]), MODE_PRINTED, tesseract_cmd=fake_tesseract.cmd)


# --------------------------------------------------------------------------- real Tesseract


@pytest.mark.integration
def test_real_tesseract_reads_generated_scan():
    try:
        pipeline.tesseract_version()
    except TesseractNotFoundError:
        pytest.skip("Tesseract is not installed (set TESSERACT_CMD to run this test)")
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 400), "white")
    ImageDraw.Draw(image).text((60, 140), "INVOICE NUMBER 4711", fill="black", font_size=90)
    result = process_pdf(image_pdf([image], page_size=(800, 200)), MODE_PRINTED, dpi=200)
    assert "INVOICE" in result.pages[0].text.upper()
    assert "4711" in result.pages[0].text
