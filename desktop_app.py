"""PDF to Word OCR — native PySide6 desktop application (entry point).

Usage:
    python desktop_app.py [document.pdf]
    python desktop_app.py --self-test report.json   # headless dependency check
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import traceback
from pathlib import Path


def _redirect_missing_std_streams() -> None:
    """Windowed (no-console) builds start with sys.stdout/sys.stderr set to None.

    Libraries that print (Hugging Face download progress bars, warnings) would
    then crash, so send their output to a log file instead.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    from ocr_app.utils import user_data_dir

    try:
        log_dir = user_data_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = open(log_dir / "app.log", "a", encoding="utf-8", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


_redirect_missing_std_streams()

from PySide6.QtCore import QObject, QSettings, QThread, Signal, Slot  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ocr_app import __version__  # noqa: E402
from ocr_app.docx_export import ExportError, export_docx  # noqa: E402
from ocr_app.pipeline import (  # noqa: E402
    DEFAULT_DPI,
    MODE_AUTO,
    MODE_HANDWRITING,
    MODE_PRINTED,
    MODES,
    OCRError,
    SUPPORTED_EXTENSIONS,
    ProcessingCancelled,
    process_pdf,
)
from ocr_app.utils import APP_NAME  # noqa: E402

FILE_FILTER = "PDFs and images ({});;All files (*)".format(" ".join("*" + e for e in SUPPORTED_EXTENSIONS))


def is_supported_file(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_EXTENSIONS)

MODE_HINTS = {
    MODE_AUTO: "Uses a PDF's embedded text where present and Tesseract OCR for scans and images. "
    "Auto does not switch to handwriting recognition — choose Handwriting for handwritten pages "
    "(pages that OCR with low confidence get a warning).",
    MODE_PRINTED: "Runs Tesseract OCR on every page, ignoring any embedded text. "
    "Best for scanned typed or printed documents.",
    MODE_HANDWRITING: "Runs the TrOCR handwriting model (English) on every page, line by line. "
    "The model (~1.3 GB) is downloaded on first use and then cached. Slower than printed OCR.",
}


class OcrWorker(QObject):
    """Runs the OCR pipeline in a background QThread and reports via signals only."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)  # DocumentResult
    failed = Signal(str, str)  # user message, technical details
    cancelled = Signal()

    def __init__(self, pdf_path: str, options: dict):
        super().__init__()
        self.pdf_path = pdf_path
        self.options = options
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Thread-safe; called from the GUI thread."""
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        try:
            result = process_pdf(
                self.pdf_path,
                status_callback=self.status.emit,
                progress_callback=self.progress.emit,
                cancel_check=self._cancel.is_set,
                **self.options,
            )
        except ProcessingCancelled:
            self.cancelled.emit()
        except OCRError as exc:
            self.failed.emit(str(exc), "")
        except Exception as exc:  # unexpected: report with traceback rather than crash
            self.failed.emit(f"Unexpected error: {exc}", traceback.format_exc())
        else:
            self.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.settings = settings if settings is not None else QSettings(APP_NAME, APP_NAME)
        self.setWindowTitle(f"PDF to Word OCR {__version__}")
        self.resize(960, 780)
        self.setAcceptDrops(True)
        self.pdf_path: str | None = None
        self._auto_output = True
        self._thread: QThread | None = None
        self._worker: OcrWorker | None = None
        self._close_requested = False
        self.last_result = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        title = QLabel("PDF to Word OCR")
        title.setObjectName("title")
        subtitle = QLabel(
            "Convert PDFs and images — digital, scanned or handwritten — into editable Word documents. "
            "All processing happens locally on this computer."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        file_row = QHBoxLayout()
        self.choose_btn = QPushButton("Choose file…")
        self.choose_btn.clicked.connect(self.choose_pdf)
        self.file_label = QLabel("No file selected — choose a PDF or image (PNG, JPG, TIFF…), or drop one onto this window")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)
        file_row.addWidget(self.choose_btn)
        file_row.addWidget(self.file_label, 1)
        layout.addLayout(file_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Save Word file as"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output .docx path")
        self.output_edit.textEdited.connect(self._output_edited)
        self.output_btn = QPushButton("Browse…")
        self.output_btn.clicked.connect(self.choose_output)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(self.output_btn)
        layout.addLayout(out_row)

        options = QHBoxLayout()
        options.addWidget(QLabel("Mode"))
        self.mode = QComboBox()
        self.mode.addItems(list(MODES))
        options.addWidget(self.mode)
        options.addSpacing(14)
        options.addWidget(QLabel("Language"))
        self.language = QComboBox()
        self.language.setEditable(True)
        self.language.addItems(["eng", "fra", "eng+fra", "spa", "deu"])
        self.language.setToolTip("Tesseract language code(s); the matching language data must be installed.")
        options.addWidget(self.language)
        options.addSpacing(14)
        options.addWidget(QLabel("DPI"))
        self.dpi = QComboBox()
        self.dpi.addItems(["150", "200", "250", "300", "350", "400"])
        self.dpi.setToolTip("Rendering resolution for OCR. 250–300 suits most scans.")
        options.addWidget(self.dpi)
        options.addSpacing(14)
        self.join_lines = QCheckBox("Join lines into paragraphs")
        self.join_lines.setToolTip(
            "Rebuild paragraphs from the printed line breaks and repair words split with a hyphen,\n"
            "so the Word text reflows when edited. Turn off to keep every original line\n"
            "(poems, addresses, lists). Handwriting always keeps one line per written line."
        )
        options.addWidget(self.join_lines)
        options.addStretch(1)
        layout.addLayout(options)

        self.mode_hint = QLabel()
        self.mode_hint.setObjectName("hint")
        self.mode_hint.setWordWrap(True)
        layout.addWidget(self.mode_hint)

        action_row = QHBoxLayout()
        self.process_btn = QPushButton("Extract text")
        self.process_btn.setObjectName("primary")
        self.process_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.export_btn = QPushButton("Save Word document")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.save_docx)
        action_row.addWidget(self.process_btn)
        action_row.addWidget(self.cancel_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.export_btn)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        preview_label = QLabel("Extracted text — review and correct OCR mistakes before saving")
        preview_label.setObjectName("sectionTitle")
        layout.addWidget(preview_label)

        self.preview = QPlainTextEdit()
        self.preview.setPlaceholderText(
            "Recognized text appears here and can be edited. Lines like '===== PAGE 2 · … =====' "
            "mark page boundaries and become page breaks in Word."
        )
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.preview.textChanged.connect(self._update_export_enabled)
        layout.addWidget(self.preview, 1)

        note = QLabel(
            "OCR can misread handwriting, names, numbers, tables and equations; layout, images and "
            "formatting are not preserved. Always proofread important content."
        )
        note.setObjectName("footnote")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.mode.currentTextChanged.connect(self._mode_changed)
        self._load_settings()
        self._mode_changed(self.mode.currentText())
        self._apply_style()

    # ------------------------------------------------------------------ settings

    def _load_settings(self) -> None:
        s = self.settings
        mode = s.value("mode", MODE_AUTO, type=str)
        self.mode.setCurrentText(mode if mode in MODES else MODE_AUTO)
        self.language.setCurrentText(s.value("language", "eng", type=str) or "eng")
        dpi = s.value("dpi", str(DEFAULT_DPI), type=str)
        self.dpi.setCurrentText(dpi if self.dpi.findText(dpi) >= 0 else str(DEFAULT_DPI))
        self.join_lines.setChecked(s.value("join_lines", True, type=bool))

    def save_settings(self) -> None:
        s = self.settings
        s.setValue("mode", self.mode.currentText())
        s.setValue("language", self.language.currentText().strip())
        s.setValue("dpi", self.dpi.currentText())
        s.setValue("join_lines", self.join_lines.isChecked())
        s.sync()

    def processing_options(self) -> dict:
        # Engines use their defaults: Tesseract is found automatically (a "tesseract" folder
        # beside the app, TESSERACT_CMD, PATH or the standard install folder) and the default
        # handwriting model is loaded from beside the app or the local cache, downloaded once
        # if missing.
        return {
            "mode": self.mode.currentText(),
            "language": self.language.currentText().strip() or "eng",
            "dpi": int(self.dpi.currentText()),
            "join_lines": self.join_lines.isChecked(),
        }

    # ------------------------------------------------------------------ dialogs (overridable in tests)

    def _info(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _error(self, title: str, text: str, details: str = "") -> None:
        box = QMessageBox(QMessageBox.Icon.Critical, title, text, QMessageBox.StandardButton.Ok, self)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text)
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ file selection

    def choose_pdf(self) -> None:
        start = str(Path(self.pdf_path).parent) if self.pdf_path else self.settings.value("last_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Select a PDF or image", start, FILE_FILTER)
        if path:
            self.set_pdf(path)

    def set_pdf(self, path: str) -> bool:
        if self.is_busy:
            return False
        if self.preview.document().isModified() and self.preview.toPlainText().strip():
            if not self._confirm("Discard edits?", "Selecting a new PDF discards the edited text. Continue?"):
                return False
        self.pdf_path = str(Path(path))
        self.settings.setValue("last_dir", str(Path(path).parent))
        self.file_label.setText(self.pdf_path)
        if self._auto_output or not self.output_edit.text().strip():
            p = Path(self.pdf_path)
            self.output_edit.setText(str(p.with_name(p.stem + "_ocr.docx")))
            self._auto_output = True
        self.preview.clear()
        self.preview.document().setModified(False)
        self.last_result = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("PDF selected. Choose a mode and click Extract text.")
        return True

    def _output_edited(self, _text: str) -> None:
        self._auto_output = False

    def choose_output(self) -> None:
        start = self.output_edit.text().strip() or self.settings.value("last_dir", "", type=str)
        path, _ = QFileDialog.getSaveFileName(self, "Save Word document as", start, "Word document (*.docx)")
        if path:
            if not path.lower().endswith(".docx"):
                path += ".docx"
            self.output_edit.setText(str(Path(path)))
            self._auto_output = False

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(is_supported_file(u.toLocalFile()) for u in urls) and not self.is_busy:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt override)
        for url in event.mimeData().urls():
            if is_supported_file(url.toLocalFile()):
                self.set_pdf(url.toLocalFile())
                break

    # ------------------------------------------------------------------ processing

    @property
    def is_busy(self) -> bool:
        return self._thread is not None

    def _mode_changed(self, mode: str) -> None:
        self.mode_hint.setText(MODE_HINTS.get(mode, ""))
        self.language.setEnabled(mode != MODE_HANDWRITING)

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.choose_btn, self.process_btn, self.mode, self.language, self.dpi,
            self.join_lines, self.preview,
        ):
            widget.setEnabled(not busy)
        if not busy:
            self.language.setEnabled(self.mode.currentText() != MODE_HANDWRITING)
        self.cancel_btn.setEnabled(busy)
        self._update_export_enabled()

    def _update_export_enabled(self) -> None:
        self.export_btn.setEnabled(not self.is_busy and bool(self.preview.toPlainText().strip()))

    def start_processing(self) -> bool:
        if self.is_busy:
            return False
        if not self.pdf_path:
            self._info("Choose a file", "Please choose a PDF or image file first.")
            return False
        if not Path(self.pdf_path).is_file():
            self._warn("File not found", f"The selected file no longer exists:\n{self.pdf_path}")
            return False
        if self.preview.document().isModified() and self.preview.toPlainText().strip():
            if not self._confirm("Replace edited text?", "Extracting again replaces the edited text. Continue?"):
                return False
        self.save_settings()

        thread = QThread()
        worker = OcrWorker(self.pdf_path, self.processing_options())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status.connect(self._on_status)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        for signal in (worker.finished, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        # Keep Python references until the thread has really finished.
        self._thread, self._worker = thread, worker

        self._set_busy(True)
        self.progress.setRange(0, 0)  # indeterminate until the page count is known
        self.status_label.setText("Starting…")
        thread.start()
        return True

    def cancel_processing(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelling after the current step…")

    # Worker-signal handlers. Signals emitted in the worker thread are queued to
    # this window's (GUI) thread. These are deliberately plain methods, not
    # @Slot: with PySide6 6.12 a @Slot overridden in a subclass is invoked in
    # the emitting thread even for queued connections, while plain methods are
    # always delivered to the receiver's thread (covered by tests/test_gui.py).

    def _on_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_finished(self, result) -> None:
        self.last_result = result
        self.preview.setPlainText(result.to_text())
        self.preview.document().setModified(False)
        message = f"Finished: {len(result.pages)} page(s) processed."
        warnings = result.warnings
        if warnings:
            self.status_label.setText(f"{message} {len(warnings)} warning(s).")
            shown = "\n".join(warnings[:15]) + ("\n…" if len(warnings) > 15 else "")
            self._warn("Finished with warnings", shown)
        else:
            self.status_label.setText(f"{message} Review the text, then save it as a Word document.")

    def _on_failed(self, message: str, details: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Processing failed: " + message.splitlines()[0])
        self._error("Processing failed", message, details)

    def _on_cancelled(self) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Cancelled.")

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
        self._set_busy(False)
        if self._close_requested:
            self.close()

    # ------------------------------------------------------------------ export

    def save_docx(self) -> bool:
        text = self.preview.toPlainText()
        if not text.strip():
            self._info("No text", "Extract text from a PDF first.")
            return False
        path = self.output_edit.text().strip()
        if not path:
            self.choose_output()
            path = self.output_edit.text().strip()
            if not path:
                return False
        target = Path(path)
        if target.suffix.lower() != ".docx":
            target = target.with_name(target.name + ".docx")
        if target.exists() and not self._confirm("Overwrite file?", f"{target} already exists. Replace it?"):
            return False
        try:
            written = export_docx(text, target, title=Path(self.pdf_path).stem if self.pdf_path else None)
        except ExportError as exc:
            self._error("Save failed", str(exc))
            return False
        except Exception as exc:
            self._error("Save failed", f"Unexpected error: {exc}", traceback.format_exc())
            return False
        self.output_edit.setText(str(written))
        self.status_label.setText(f"Saved Word document: {written}")
        return True

    # ------------------------------------------------------------------ shutdown

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self.is_busy:
            if self._close_requested or self._confirm(
                "Processing in progress", "Cancel processing and quit when the current step finishes?"
            ):
                self._close_requested = True
                self.cancel_processing()
            event.ignore()
            return
        self.save_settings()
        event.accept()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #f6f8fb; }
            QWidget { color: #202938; font-size: 10pt; }
            QLabel#title { font-size: 21pt; font-weight: 700; color: #152238; }
            QLabel#subtitle { color: #5d6879; font-size: 10.5pt; margin-bottom: 4px; }
            QLabel#fileLabel { color: #465368; }
            QLabel#hint { color: #5d6879; font-size: 9pt; }
            QLabel#sectionTitle { font-weight: 600; font-size: 11pt; margin-top: 4px; }
            QLabel#status { color: #42526a; }
            QLabel#footnote { color: #697586; font-size: 9pt; }
            QPushButton { background: white; border: 1px solid #cbd4e1; border-radius: 7px; padding: 8px 14px; }
            QPushButton:hover { background: #edf3fa; }
            QPushButton:disabled { color: #9aa4b2; background: #eef1f5; }
            QPushButton#primary { background: #2458d3; color: white; border: 1px solid #2458d3; font-weight: 600; }
            QPushButton#primary:hover { background: #1e49b2; }
            QPushButton#primary:disabled { background: #9db4ea; border-color: #9db4ea; }
            QComboBox, QLineEdit { background: white; border: 1px solid #cbd4e1; border-radius: 6px; padding: 6px; min-width: 70px; }
            QPlainTextEdit { background: white; border: 1px solid #cbd4e1; border-radius: 8px; padding: 10px; selection-background-color: #b9d1ff; }
            QProgressBar { border: none; background: #e3e9f1; border-radius: 3px; }
            QProgressBar::chunk { background: #2458d3; border-radius: 3px; }
        """)


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    argv = list(sys.argv if argv is None else argv)
    if "--self-test" in argv:
        index = argv.index("--self-test")
        report = argv[index + 1] if len(argv) > index + 1 else None
        from ocr_app.selftest import run_self_test

        return run_self_test(report, window_factory=MainWindow)

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("PDF to Word OCR")
    app.setOrganizationName(APP_NAME)
    window = MainWindow()
    pdf_args = [a for a in argv[1:] if is_supported_file(a)]
    if pdf_args:
        window.set_pdf(pdf_args[0])
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
