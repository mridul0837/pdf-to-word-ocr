"""OCR pipeline: PDF text extraction, printed OCR (Tesseract) and handwriting (TrOCR).

This module has no GUI dependencies. Heavy optional dependencies (pytesseract,
OpenCV, PyTorch, Transformers) are imported lazily so that a missing package or
model only produces an error when the feature that needs it is actually used.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import pymupdf
from PIL import Image

from .utils import app_base_dir, clean_text, page_marker

MODE_AUTO = "Auto"
MODE_PRINTED = "Printed Text"
MODE_HANDWRITING = "Handwriting"
MODES = (MODE_AUTO, MODE_PRINTED, MODE_HANDWRITING)

METHOD_EMBEDDED = "Embedded PDF text"
METHOD_TESSERACT = "Tesseract OCR"
METHOD_TROCR = "TrOCR handwriting"
METHOD_FAILED = "Failed"

DEFAULT_HANDWRITING_MODEL = "microsoft/trocr-base-handwritten"
DEFAULT_DPI = 250
MIN_DPI, MAX_DPI = 72, 600
# Pages are rendered at most this large; protects against poster-sized pages.
MAX_RENDER_PIXELS = 40_000_000
# Embedded text with at least this many non-space characters is trusted in Auto mode.
MIN_EMBEDDED_CHARS = 20

TESSERACT_EXE = "tesseract.exe" if os.name == "nt" else "tesseract"
_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$")

PdfSource = Union[bytes, bytearray, memoryview, str, os.PathLike]
StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]


# --------------------------------------------------------------------------- errors


class OCRError(Exception):
    """An error with a message suitable for showing to the user."""


class PDFOpenError(OCRError):
    """The input is missing, empty, damaged, encrypted or not a PDF."""


class TesseractNotFoundError(OCRError):
    """The Tesseract executable (or pytesseract) is not available."""


class TesseractLanguageError(OCRError):
    """The requested Tesseract language data is not installed."""


class HandwritingModelError(OCRError):
    """The handwriting model or its libraries could not be loaded."""


class ProcessingCancelled(Exception):
    """Raised when the caller's cancel check returns True."""


# --------------------------------------------------------------------------- results


@dataclass
class PageResult:
    number: int  # 1-based page number
    text: str
    method: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class DocumentResult:
    pages: list[PageResult]
    mode: str

    @property
    def warnings(self) -> list[str]:
        return [f"Page {p.number}: {w}" for p in self.pages for w in p.warnings]

    def to_text(self) -> str:
        """Editable preview text with a marker line at each page boundary."""
        return "\n\n".join(
            f"{page_marker(p.number, p.method)}\n{p.text}".rstrip() for p in self.pages
        )


def normalize_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    aliases = {
        "auto": MODE_AUTO,
        "printed": MODE_PRINTED,
        "printed text": MODE_PRINTED,
        "handwriting": MODE_HANDWRITING,
        "handwritten": MODE_HANDWRITING,
    }
    if key not in aliases:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    return aliases[key]


# --------------------------------------------------------------------------- PDF


def open_pdf(source: PdfSource) -> pymupdf.Document:
    """Open a PDF from a path or bytes, raising PDFOpenError with a readable message."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data, name = bytes(source), "The PDF data"
    else:
        path = Path(source)
        name = path.name or str(path)
        if not path.is_file():
            raise PDFOpenError(f"File not found: {path}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PDFOpenError(f"Could not read {name}: {exc}") from exc
    if not data.strip():
        raise PDFOpenError(f"{name} is empty (0 bytes).")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types here
        raise PDFOpenError(f"{name} is not a valid PDF or is damaged ({exc}).") from exc
    if not doc.is_pdf:  # MuPDF sniffs content and may open e.g. image data despite filetype="pdf"
        doc.close()
        raise PDFOpenError(f"{name} is not a valid PDF (it looks like a different file type).")
    if doc.needs_pass:
        doc.close()
        raise PDFOpenError(f"{name} is password-protected. Remove the password and try again.")
    if doc.page_count == 0:
        doc.close()
        raise PDFOpenError(f"{name} contains no readable pages (the file may be damaged).")
    return doc


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    """Render a page to an RGB PIL image, capping very large pages."""
    zoom = dpi / 72.0
    width_in, height_in = page.rect.width / 72.0, page.rect.height / 72.0
    pixels = (width_in * dpi) * (height_in * dpi)
    if pixels > MAX_RENDER_PIXELS:
        zoom *= (MAX_RENDER_PIXELS / pixels) ** 0.5
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    return Image.frombuffer("RGB", (pix.width, pix.height), pix.samples, "raw", "RGB", pix.stride, 1)


def _non_space_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


# --------------------------------------------------------------------------- Tesseract


TESSERACT_HELP = (
    "Tesseract OCR was not found. Install it (for example the UB Mannheim build from "
    "https://github.com/UB-Mannheim/tesseract/wiki) or set the path to tesseract.exe "
    "in the app's Settings, or via the TESSERACT_CMD environment variable."
)


def _default_tesseract_locations() -> list[Path]:
    locations = [app_base_dir() / "tesseract" / TESSERACT_EXE]
    if os.name == "nt":
        for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            base = os.environ.get(var)
            if base:
                locations.append(Path(base) / "Tesseract-OCR" / TESSERACT_EXE)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            locations.append(Path(local) / "Programs" / "Tesseract-OCR" / TESSERACT_EXE)
            locations.append(Path(local) / "Tesseract-OCR" / TESSERACT_EXE)
    return locations


def find_tesseract(configured: Optional[str] = None) -> str:
    """Return the Tesseract executable path.

    Search order: explicitly configured path, TESSERACT_CMD, a ``tesseract``
    folder next to the app, PATH, then the standard Windows install folders.
    """
    if configured and configured.strip():
        path = Path(configured.strip().strip('"')).expanduser()
        if path.is_dir():
            path = path / TESSERACT_EXE
        if path.is_file():
            return str(path)
        raise TesseractNotFoundError(
            f"The configured Tesseract path does not exist: {configured}\n\n{TESSERACT_HELP}"
        )
    env = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    if env and Path(env).is_file():
        return env
    bundled = _default_tesseract_locations()
    if bundled[0].is_file():
        return str(bundled[0])
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in bundled[1:]:
        if candidate.is_file():
            return str(candidate)
    raise TesseractNotFoundError(TESSERACT_HELP)


def _import_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise TesseractNotFoundError(
            "The 'pytesseract' Python package is not installed (pip install pytesseract)."
        ) from exc
    return pytesseract


def tesseract_version(configured: Optional[str] = None) -> str:
    """Locate Tesseract and return its version string (raises TesseractNotFoundError)."""
    pytesseract = _import_pytesseract()
    cmd = find_tesseract(configured)
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception as exc:
        raise TesseractNotFoundError(f"Tesseract at {cmd} could not be run: {exc}") from exc


class TesseractOCR:
    """Printed-text OCR via the Tesseract executable.

    The executable and language data are validated on first use; a failure is
    remembered so later pages fail fast with the same message.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None, language: str = "eng", psm: int = 3):
        language = (language or "eng").strip()
        if not _LANGUAGE_RE.match(language):
            raise TesseractLanguageError(
                f"Invalid Tesseract language {language!r}. Use codes like 'eng' or 'eng+fra'."
            )
        self.configured_cmd = tesseract_cmd
        self.language = language
        self.psm = psm
        self.cmd: Optional[str] = None
        self.error: Optional[OCRError] = None

    def ensure_ready(self) -> None:
        if self.cmd is not None:
            return
        if self.error is not None:
            raise self.error
        try:
            pytesseract = _import_pytesseract()
            cmd = find_tesseract(self.configured_cmd)
            pytesseract.pytesseract.tesseract_cmd = cmd
            try:
                available = set(pytesseract.get_languages(config=""))
            except Exception as exc:
                raise TesseractNotFoundError(f"Tesseract at {cmd} could not be run: {exc}") from exc
            if not available:
                raise TesseractLanguageError(
                    f"Tesseract at {cmd} found no language data. Put the language files (for "
                    "example eng.traineddata) in its 'tessdata' folder, or set the TESSDATA_PREFIX "
                    "environment variable to the folder that contains them."
                )
            missing = [code for code in self.language.split("+") if code not in available]
            if missing:
                raise TesseractLanguageError(
                    f"Tesseract language data not installed: {', '.join(missing)}. "
                    f"Installed: {', '.join(sorted(available)) or 'none'}. Add the matching "
                    f".traineddata files to Tesseract's 'tessdata' folder."
                )
        except OCRError as exc:
            self.error = exc
            raise
        self.cmd = cmd

    def recognize(self, image: Image.Image) -> str:
        self.ensure_ready()
        pytesseract = _import_pytesseract()
        pytesseract.pytesseract.tesseract_cmd = self.cmd
        try:
            return pytesseract.image_to_string(image, lang=self.language, config=f"--psm {self.psm}")
        except pytesseract.TesseractNotFoundError as exc:
            raise TesseractNotFoundError(f"Tesseract at {self.cmd} could not be run.") from exc
        except pytesseract.TesseractError as exc:
            raise OCRError(f"Tesseract failed: {exc.message or exc}") from exc


# --------------------------------------------------------------------------- handwriting


Box = tuple[int, int, int, int]  # x, y, width, height


def segment_lines(image: Image.Image) -> list[Box]:
    """Find text lines on a page with a simple OpenCV heuristic, top to bottom.

    Returns an empty list for blank / near-uniform pages so that the
    handwriting model is never asked to read an empty crop (TrOCR tends to
    hallucinate text for those).
    """
    import cv2
    import numpy as np

    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return []
    gray = cv2.medianBlur(gray, 3)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binary > 0
    min_ink_pixels = max(50, int(gray.size * 0.00005))
    if int(ink.sum()) < min_ink_pixels or ink.all():
        return []
    contrast = float(gray[~ink].mean()) - float(gray[ink].mean())
    if contrast < 40:
        return []

    img_h, img_w = binary.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, img_w // 40), 3))
    joined = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_h, min_w = max(8, img_h // 200), max(12, img_w // 150)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[3] >= min_h and b[2] >= min_w]
    boxes.sort(key=lambda b: b[1] + b[3] / 2)

    lines: list[list[int]] = []
    for x, y, w, h in boxes:
        if lines:
            lx, ly, lw, lh = lines[-1]
            overlap = min(y + h, ly + lh) - max(y, ly)
            if overlap >= 0.5 * min(h, lh):
                nx, ny = min(lx, x), min(ly, y)
                lines[-1] = [nx, ny, max(lx + lw, x + w) - nx, max(ly + lh, y + h) - ny]
                continue
        lines.append([x, y, w, h])
    lines.sort(key=lambda b: b[1])
    return [tuple(b) for b in lines]


def crop_line(image: Image.Image, box: Box) -> Image.Image:
    x, y, w, h = box
    pad = max(6, h // 4)
    return image.crop(
        (max(0, x - pad), max(0, y - pad), min(image.width, x + w + pad), min(image.height, y + h + pad))
    ).convert("RGB")


_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")


def tidy_trocr_text(text: str) -> str:
    """TrOCR (trained on IAM) emits tokenised text like 'fox , then .'; join the punctuation."""
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text).strip()


def resolve_model_source(model_id: str) -> str:
    """Prefer a local model folder (explicit path or ``models/<name>`` beside the app)."""
    model_id = (model_id or DEFAULT_HANDWRITING_MODEL).strip()
    if Path(model_id).is_dir():
        return model_id
    bundled = app_base_dir() / "models" / model_id.rstrip("/").split("/")[-1]
    if (bundled / "config.json").is_file():
        return str(bundled)
    return model_id


def select_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class HandwritingRecognizer:
    """Line-level handwriting recognition with a TrOCR model, loaded lazily.

    Weights are taken from the local Hugging Face cache (or a local folder)
    first; they are downloaded only if missing and ``allow_download`` is set.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_HANDWRITING_MODEL,
        *,
        allow_download: bool = True,
        device: Optional[str] = None,
        processor=None,
        model=None,
        batch_size: int = 4,
        max_new_tokens: int = 128,
    ):
        self.model_id = (model_id or DEFAULT_HANDWRITING_MODEL).strip()
        self.allow_download = allow_download
        self.preferred_device = device
        self.processor = processor
        self.model = model
        self.device: Optional[str] = None
        self.batch_size = max(1, batch_size)
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None and self.device is not None

    def load(self, status_callback: Optional[StatusCallback] = None) -> None:
        status = status_callback or (lambda _msg: None)
        with self._lock:
            if self.is_loaded:
                return
            if self.processor is None or self.model is None:
                self.processor, self.model = self._load_from_pretrained(status)
            self._place_model(select_device(self.preferred_device))

    def _load_from_pretrained(self, status: StatusCallback):
        try:
            import torch  # noqa: F401  (fail early with a clear message)
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except Exception as exc:  # ImportError, or OSError from broken torch DLLs
            raise HandwritingModelError(
                "Handwriting recognition needs PyTorch and Transformers, which could not be "
                f"loaded: {exc}"
            ) from exc

        source = resolve_model_source(self.model_id)
        status(f"Loading handwriting model {self.model_id}…")
        try:
            processor = TrOCRProcessor.from_pretrained(source, local_files_only=True)
            model = VisionEncoderDecoderModel.from_pretrained(source, local_files_only=True)
        except Exception as local_exc:
            if Path(source).is_dir():
                raise HandwritingModelError(
                    f"The local model folder {source} could not be loaded: {local_exc}"
                ) from local_exc
            if not self.allow_download:
                raise HandwritingModelError(
                    f"The handwriting model '{source}' is not in the local cache and downloading "
                    "is disabled. Enable 'Allow model download' in Settings (needs internet once, "
                    "about 1.3 GB), or run: python -m ocr_app.fetch_model"
                ) from local_exc
            status(f"Downloading handwriting model {source} (about 1.3 GB, first use only)…")
            try:
                processor = TrOCRProcessor.from_pretrained(source)
                model = VisionEncoderDecoderModel.from_pretrained(source)
            except Exception as exc:
                raise HandwritingModelError(
                    f"Could not download the handwriting model '{source}'. Check your internet "
                    f"connection and the model name. Details: {exc}"
                ) from exc
        model.eval()
        return processor, model

    def _place_model(self, device: str) -> None:
        if hasattr(self.model, "to"):
            try:
                self.model = self.model.to(device)
            except Exception:
                if device == "cpu":
                    raise
                device = "cpu"
                self.model = self.model.to(device)
        self.device = device

    def recognize_lines(self, crops: Sequence[Image.Image]) -> list[str]:
        """Recognise a batch of single-line images; returns one string per crop."""
        if not crops:
            return []
        import torch

        self.load()
        pixel_values = self.processor(images=list(crops), return_tensors="pt").pixel_values
        try:
            with torch.inference_mode():
                ids = self.model.generate(pixel_values.to(self.device), max_new_tokens=self.max_new_tokens)
        except RuntimeError as exc:
            if self.device == "cpu" or "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            self._place_model("cpu")
            with torch.inference_mode():
                ids = self.model.generate(pixel_values.to("cpu"), max_new_tokens=self.max_new_tokens)
        return [tidy_trocr_text(t) for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def recognize_page(
        self,
        image: Image.Image,
        status_callback: Optional[StatusCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> str:
        status = status_callback or (lambda _msg: None)
        boxes = segment_lines(image)
        if not boxes:
            return ""
        self.load(status)
        crops = [crop_line(image, box) for box in boxes]
        lines: list[str] = []
        for start in range(0, len(crops), self.batch_size):
            if cancel_check and cancel_check():
                raise ProcessingCancelled()
            end = min(start + self.batch_size, len(crops))
            status(f"Recognizing handwritten lines {start + 1}–{end} of {len(crops)} ({self.device})…")
            lines.extend(self.recognize_lines(crops[start:end]))
        return "\n".join(line for line in lines if line)


_recognizers: dict[str, HandwritingRecognizer] = {}
_recognizers_lock = threading.Lock()


def get_handwriting_recognizer(
    model_id: str = DEFAULT_HANDWRITING_MODEL, allow_download: bool = True
) -> HandwritingRecognizer:
    """Process-wide recognizer per model, so weights are loaded at most once per session."""
    key = (model_id or DEFAULT_HANDWRITING_MODEL).strip()
    with _recognizers_lock:
        recognizer = _recognizers.get(key)
        if recognizer is None:
            recognizer = HandwritingRecognizer(key, allow_download=allow_download)
            _recognizers[key] = recognizer
        recognizer.allow_download = allow_download
        return recognizer


# --------------------------------------------------------------------------- orchestration


def process_pdf(
    source: PdfSource,
    mode: str = MODE_AUTO,
    *,
    language: str = "eng",
    dpi: int = DEFAULT_DPI,
    tesseract_cmd: Optional[str] = None,
    handwriting_model: str = DEFAULT_HANDWRITING_MODEL,
    allow_model_download: bool = True,
    printed_ocr: Optional[TesseractOCR] = None,
    handwriting_recognizer: Optional[HandwritingRecognizer] = None,
    status_callback: Optional[StatusCallback] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> DocumentResult:
    """Extract text from every page of a PDF.

    * Auto: embedded text where usable, otherwise Tesseract OCR. If Tesseract
      is unavailable, pages that need OCR get a warning instead of aborting the
      run (unless no page yields any text). Auto does not detect handwriting.
    * Printed Text: Tesseract on every rendered page.
    * Handwriting: TrOCR on every rendered page (model loaded on first use).

    Configuration problems (missing Tesseract/model, cancelled) raise; other
    per-page failures are recorded as page warnings and processing continues.
    """
    mode = normalize_mode(mode)
    dpi = max(MIN_DPI, min(MAX_DPI, int(dpi)))
    status = status_callback or (lambda _msg: None)
    progress = progress_callback or (lambda _done, _total: None)
    printed = printed_ocr or TesseractOCR(tesseract_cmd, language)

    def check_cancel():
        if cancel_check and cancel_check():
            raise ProcessingCancelled()

    status("Opening PDF…")
    doc = open_pdf(source)
    pages: list[PageResult] = []
    try:
        total = doc.page_count
        progress(0, total)
        for index in range(total):
            check_cancel()
            number = index + 1
            prefix = f"Page {number} of {total}"
            status(f"{prefix}: reading…")
            try:
                page = doc.load_page(index)
                if mode == MODE_HANDWRITING:
                    recognizer = handwriting_recognizer or get_handwriting_recognizer(
                        handwriting_model, allow_model_download
                    )
                    image = render_page(page, dpi)
                    text = recognizer.recognize_page(
                        image, lambda m: status(f"{prefix}: {m}"), cancel_check
                    )
                    result = PageResult(number, clean_text(text), METHOD_TROCR)
                elif mode == MODE_PRINTED:
                    status(f"{prefix}: running Tesseract OCR…")
                    text = printed.recognize(render_page(page, dpi))
                    result = PageResult(number, clean_text(text), METHOD_TESSERACT)
                else:
                    result = _process_auto_page(page, number, dpi, printed, status, prefix)
            except (ProcessingCancelled, HandwritingModelError, TesseractNotFoundError, TesseractLanguageError):
                raise
            except Exception as exc:
                result = PageResult(number, "", METHOD_FAILED, [f"Processing failed: {exc}"])
            if not result.text and not result.warnings:
                result.warnings.append("No text recognized (blank page?).")
            pages.append(result)
            progress(number, total)
    finally:
        doc.close()

    # Auto mode tolerates a missing Tesseract only if something was still extracted.
    if printed.error is not None and not any(p.text for p in pages):
        raise printed.error
    return DocumentResult(pages=pages, mode=mode)


def _process_auto_page(page, number, dpi, printed: TesseractOCR, status, prefix) -> PageResult:
    embedded = clean_text(page.get_text("text"))
    has_images = bool(page.get_images(full=False))
    if _non_space_len(embedded) >= MIN_EMBEDDED_CHARS or (embedded and not has_images):
        return PageResult(number, embedded, METHOD_EMBEDDED)

    status(f"{prefix}: no usable embedded text, running Tesseract OCR…")
    try:
        ocr_text = clean_text(printed.recognize(render_page(page, dpi)))
    except (TesseractNotFoundError, TesseractLanguageError) as exc:
        first_line = str(exc).splitlines()[0]
        warning = f"Page needs OCR but Tesseract is unavailable: {first_line}"
        if embedded:
            return PageResult(number, embedded, METHOD_EMBEDDED, [warning])
        return PageResult(number, "", METHOD_TESSERACT, [warning])
    if not ocr_text and embedded:
        return PageResult(number, embedded, METHOD_EMBEDDED + " (OCR found nothing)")
    return PageResult(number, ocr_text, METHOD_TESSERACT)
