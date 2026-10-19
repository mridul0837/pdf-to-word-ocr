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


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        process_pdf(digital_pdf(["text"]), mode="Magic")
