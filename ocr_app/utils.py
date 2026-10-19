"""Small helpers shared by the OCR pipeline, the DOCX exporter and the GUI."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

APP_NAME = "PDFtoWordOCR"

# Characters that are not allowed in XML 1.0 (and therefore in .docx files):
# C0 control characters other than tab/newline/carriage return, DEL, lone
# surrogates and the two non-characters U+FFFE / U+FFFF. PDF text layers and
# Tesseract output (form feed after every page) regularly contain these.
_XML_ILLEGAL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff￾￿]")
_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Page boundary lines shown in the editable preview, e.g.
# "===== PAGE 3 · Tesseract OCR =====". The exporter turns them into page breaks.
PAGE_MARKER_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\b.*=====\s*$")


def app_base_dir() -> Path:
    """Directory holding the application: the .exe folder when frozen, else the project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Per-user writable directory for logs (not created here)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def xml_safe(text: str) -> str:
    """Remove characters that cannot be stored in a Word document."""
    return _XML_ILLEGAL_RE.sub("", text)


def clean_text(text: str | None) -> str:
    """Normalise newlines, drop illegal characters, trailing spaces and excess blank lines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = xml_safe(text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def page_marker(number: int, method: str) -> str:
    return f"===== PAGE {number} · {method} ====="


def is_page_marker(line: str) -> bool:
    return bool(PAGE_MARKER_RE.match(line.strip()))
