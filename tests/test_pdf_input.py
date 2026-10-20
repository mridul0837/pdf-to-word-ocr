"""Invalid, malformed, empty and protected PDF input."""

import pymupdf
import pytest

from conftest import digital_pdf
from ocr_app.pipeline import PDFOpenError, open_pdf, process_pdf


def test_opens_valid_pdf_from_path_and_bytes(tmp_path):
    data = digital_pdf(["Hello from a valid PDF"])
    path = tmp_path / "ok.pdf"
    path.write_bytes(data)
    for source in (data, path, str(path)):
        doc = open_pdf(source)
        assert doc.page_count == 1
        doc.close()


@pytest.mark.parametrize(
    "payload",
    [b"", b"   \n\t  "],
    ids=["zero-bytes", "whitespace-only"],
)
def test_empty_input_is_rejected(payload):
    with pytest.raises(PDFOpenError, match="empty"):
        open_pdf(payload)


def test_empty_file_on_disk_is_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(PDFOpenError, match="empty.pdf is empty"):
        process_pdf(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(PDFOpenError, match="File not found"):
        process_pdf(tmp_path / "does_not_exist.pdf")


@pytest.mark.parametrize(
    "payload",
    [
        b"This is plainly not a PDF document. " * 20,
        b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj << /Type /Catalog",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 200,
    ],
    ids=["text", "header-only", "png-bytes"],
)
def test_malformed_input_is_rejected(payload):
    with pytest.raises(PDFOpenError, match="not a valid PDF"):
        process_pdf(payload)


def test_truncated_pdf_is_rejected():
    data = digital_pdf(["Some content that will be cut off"])
    with pytest.raises(PDFOpenError):
        process_pdf(data[: len(data) // 3])


def test_password_protected_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    with pytest.raises(PDFOpenError, match="password-protected"):
        process_pdf(data)


def image_bytes(fmt, size=(600, 800), frames=1):
    import io

    from PIL import Image

    images = [Image.new("RGB", size, (255, 255 - 50 * i, 255)) for i in range(frames)]
    buf = io.BytesIO()
    images[0].save(buf, format=fmt, save_all=frames > 1, append_images=images[1:])
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "GIF", "TIFF", "WEBP"])
def test_images_are_accepted_as_one_page(fmt):
    doc = open_pdf(image_bytes(fmt))
    assert doc.is_pdf and doc.page_count == 1
    pix = doc[0].get_pixmap(dpi=96)  # images without DPI metadata are treated as 96 DPI
    assert (pix.width, pix.height) == (600, 800)
    doc.close()


def test_multipage_tiff_gives_one_page_per_frame():
    doc = open_pdf(image_bytes("TIFF", frames=3))
    assert doc.page_count == 3
    doc.close()


def test_png_saved_with_pdf_extension_is_processed(tmp_path, fake_tesseract):
    # A real case: a scanned page saved as PNG but named "output-1.pdf".
    from conftest import png_bytes, text_image

    path = tmp_path / "output-1.pdf"
    path.write_bytes(png_bytes(text_image("scanned")))
    result = process_pdf(path, "Printed Text", tesseract_cmd=fake_tesseract.cmd)
    assert [p.text for p in result.pages] == ["Recognized printed text 1"]


def test_damaged_image_is_rejected():
    with pytest.raises(PDFOpenError, match="not a valid PDF or image"):
        process_pdf(image_bytes("JPEG")[:300])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        process_pdf(digital_pdf(["text"]), mode="Magic")
