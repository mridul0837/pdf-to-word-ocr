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
    assert window.allow_download.isChecked()
    assert not window.export_btn.isEnabled()
    assert not window.cancel_btn.isEnabled()
    assert window.process_btn.isEnabled()
    assert "does not detect handwriting" in window.mode_hint.text()


def test_handwriting_mode_disables_language(window):
    window.mode.setCurrentText(MODE_HANDWRITING)
    assert not window.language.isEnabled()
    assert "TrOCR" in window.mode_hint.text()
    window.mode.setCurrentText(MODE_AUTO)
    assert window.language.isEnabled()


def test_extract_without_pdf_shows_message(window):
    assert window.start_processing() is False
    assert window.dialogs == [("info", "Choose a PDF", "Please choose a PDF file first.")]


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
    first.tesseract_edit.setText(r"C:\Tools\Tesseract-OCR\tesseract.exe")
    first.allow_download.setChecked(False)
    first.dpi.setCurrentText("300")
    first.save_settings()
    first.deleteLater()

    second = RecordingWindow(QSettings(ini, QSettings.Format.IniFormat))
    assert second.mode.currentText() == MODE_HANDWRITING
    assert second.tesseract_edit.text() == r"C:\Tools\Tesseract-OCR\tesseract.exe"
    assert not second.allow_download.isChecked()
    assert second.processing_options()["dpi"] == 300
    assert second.processing_options()["allow_model_download"] is False
    second.deleteLater()


def test_missing_tesseract_test_button_reports_problem(window, no_tesseract):
    window.tesseract_edit.setText("")
    window.test_tesseract()
    assert window.dialogs[-1][0] == "warn"
    assert "Tesseract OCR was not found" in window.dialogs[-1][2]
