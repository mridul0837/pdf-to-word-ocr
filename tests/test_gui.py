"""GUI tests on the offscreen Qt platform (no real dialogs, no network, no OCR engines)."""

import time

import pytest
from docx import Document
from PySide6.QtCore import QSettings, QThread
from PySide6.QtWidgets import QApplication

import desktop_app
from conftest import digital_pdf
from ocr_app.pipeline import MODE_AUTO, MODE_HANDWRITING, MODES, ProcessingCancelled


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


class RecordingWindow(desktop_app.MainWindow):
    """Records dialogs instead of showing them, and the threads that run worker-signal handlers."""

    def __init__(self, settings):
        super().__init__(settings)
        self.dialogs = []
        self.handler_threads = []

    def _info(self, title, text):
        self.dialogs.append(("info", title, text))

    def _warn(self, title, text):
        self.dialogs.append(("warn", title, text))

    def _error(self, title, text, details=""):
        self.dialogs.append(("error", title, text))

    def _confirm(self, title, text):
        self.dialogs.append(("confirm", title, text))
        return True

    def _on_status(self, message):
        self.handler_threads.append(("status", QThread.currentThread()))
        super()._on_status(message)

    def _on_progress(self, done, total):
        self.handler_threads.append(("progress", QThread.currentThread()))
        super()._on_progress(done, total)

    def _on_finished(self, result):
        self.handler_threads.append(("finished", QThread.currentThread()))
        super()._on_finished(result)


@pytest.fixture
def window(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = RecordingWindow(settings)
    yield win
    if win.is_busy:
        win.cancel_processing()
        wait_until(qapp, lambda: not win.is_busy)
    win.close()
    win.deleteLater()
    qapp.processEvents()


def wait_until(qapp, predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the GUI")


def test_window_initial_state(window):
    assert window.windowTitle().startswith("PDF to Word OCR")
    assert [window.mode.itemText(i) for i in range(window.mode.count())] == list(MODES)
    assert window.mode.currentText() == MODE_AUTO
    assert window.dpi.currentText() == "250"
    # No engine settings in the GUI: Tesseract path, model and download use the defaults.
    assert window.processing_options() == {"mode": MODE_AUTO, "language": "eng", "dpi": 250, "join_lines": True}
    assert not window.export_btn.isEnabled()
    assert not window.cancel_btn.isEnabled()
    assert window.process_btn.isEnabled()
    assert "does not switch to handwriting" in window.mode_hint.text()


def test_handwriting_mode_disables_language(window):
    window.mode.setCurrentText(MODE_HANDWRITING)
    assert not window.language.isEnabled()
    assert "TrOCR" in window.mode_hint.text()
    window.mode.setCurrentText(MODE_AUTO)
    assert window.language.isEnabled()


def test_extract_without_pdf_shows_message(window):
    assert window.start_processing() is False
    assert window.dialogs == [("info", "Choose a file", "Please choose a PDF or image file first.")]


def test_selecting_pdf_suggests_output_path(window, tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(digital_pdf(["x"]))
    window.set_pdf(str(pdf))
    assert window.output_edit.text() == str(tmp_path / "scan_ocr.docx")


def test_full_run_in_background_thread_then_export(window, qapp, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(digital_pdf(["Hello from page one of the GUI test.", "And this is page two of it."]))
    window.set_pdf(str(pdf))
    assert window.start_processing() is True
    assert window.is_busy and window.cancel_btn.isEnabled() and not window.process_btn.isEnabled()
    wait_until(qapp, lambda: not window.is_busy)

    # Every worker signal was handled in the GUI thread, never in the worker thread.
    kinds = [kind for kind, _thread in window.handler_threads]
    assert {"status", "progress", "finished"} <= set(kinds) and kinds[-1] == "finished"
    assert all(thread is qapp.thread() for _kind, thread in window.handler_threads)
    text = window.preview.toPlainText()
    assert "===== PAGE 1 · Embedded PDF text =====" in text
    assert text.index("page one") < text.index("page two")
    assert window.status_label.text().startswith("Finished: 2 page(s)")
    assert window.export_btn.isEnabled() and window.process_btn.isEnabled()
    assert window.progress.value() == window.progress.maximum() == 2

    window.preview.setPlainText(text.replace("page one", "PAGE ONE (edited)"))
    assert window.save_docx() is True
    out = tmp_path / "doc_ocr.docx"
    paragraphs = [p.text for p in Document(str(out)).paragraphs if p.text]
    assert paragraphs == ["Hello from PAGE ONE (edited) of the GUI test.", "And this is page two of it."]
    assert window.status_label.text() == f"Saved Word document: {out}"


def test_invalid_pdf_reports_error_and_recovers(window, qapp, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all " * 10)
    window.set_pdf(str(bad))
    window.start_processing()
    wait_until(qapp, lambda: not window.is_busy)
    assert window.dialogs[-1][0] == "error"
    assert "not a valid PDF" in window.dialogs[-1][2]
    assert window.status_label.text().startswith("Processing failed")
    assert window.process_btn.isEnabled() and not window.export_btn.isEnabled()


def test_cancel_stops_worker(window, qapp, tmp_path, monkeypatch):
    def slow_process(path, *, cancel_check, status_callback, **_):
        status_callback("Working…")
        while not cancel_check():
            time.sleep(0.01)
        raise ProcessingCancelled()

    monkeypatch.setattr(desktop_app, "process_pdf", slow_process)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(digital_pdf(["x"]))
    window.set_pdf(str(pdf))
    window.start_processing()
    wait_until(qapp, lambda: window.status_label.text() == "Working…")
    window.cancel_processing()
    wait_until(qapp, lambda: not window.is_busy)
    assert window.status_label.text() == "Cancelled."
    assert window.process_btn.isEnabled()


def test_settings_are_persisted(qapp, tmp_path):
    ini = str(tmp_path / "persist.ini")
    first = RecordingWindow(QSettings(ini, QSettings.Format.IniFormat))
    first.mode.setCurrentText(MODE_HANDWRITING)
    first.language.setCurrentText("fra")
    first.dpi.setCurrentText("300")
    first.join_lines.setChecked(False)
    first.save_settings()
    first.deleteLater()

    second = RecordingWindow(QSettings(ini, QSettings.Format.IniFormat))
    assert second.processing_options() == {"mode": MODE_HANDWRITING, "language": "fra", "dpi": 300, "join_lines": False}
    second.deleteLater()


def test_missing_tesseract_is_reported_with_how_to_fix(window, qapp, tmp_path, no_tesseract):
    from conftest import png_bytes, text_image

    scan = tmp_path / "scan.png"
    scan.write_bytes(png_bytes(text_image("scanned page")))
    window.set_pdf(str(scan))
    window.mode.setCurrentText("Printed Text")
    window.start_processing()
    wait_until(qapp, lambda: not window.is_busy)
    kind, title, message = window.dialogs[-1]
    assert (kind, title) == ("error", "Processing failed")
    assert "Tesseract OCR was not found" in message and "TESSERACT_CMD" in message
    assert "Settings" not in message


def test_images_are_accepted_by_picker_and_drop():
    assert desktop_app.is_supported_file(r"C:\scans\letter.PDF")
    assert desktop_app.is_supported_file(r"C:\scans\photo.jpeg")
    assert desktop_app.is_supported_file(r"C:\scans\page.tiff")
    assert not desktop_app.is_supported_file(r"C:\scans\notes.docx")
    assert "*.png" in desktop_app.FILE_FILTER and "*.pdf" in desktop_app.FILE_FILTER
