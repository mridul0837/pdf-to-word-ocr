# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the PDF to Word OCR desktop app (one-folder build).
# Build with .\build_exe.ps1 (or: python -m PyInstaller --noconfirm --clean PDFtoWordOCR.spec).
# Set PDFOCR_CONSOLE=1 to build with a console window for debugging start-up errors.
import os

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# transformers imports model code lazily by name (AutoImageProcessor/AutoTokenizer/
# AutoModel), which PyInstaller cannot see. Bundle the architectures TrOCR uses:
# ViT/DeiT encoders, TrOCR decoder, RoBERTa/XLM-R tokenizers.
hiddenimports = ["pytesseract"]
for name in ("auto", "trocr", "vision_encoder_decoder", "vit", "deit", "roberta", "xlm_roberta"):
    hiddenimports += collect_submodules(f"transformers.models.{name}")

# transformers checks installed versions through package metadata at import time.
datas = []
for dist in ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy", "tqdm",
             "regex", "requests", "packaging", "filelock", "pyyaml", "pytesseract"):
    datas += copy_metadata(dist)

excludes = [
    # Optional PyMuPDF 2.x add-ons (layout analysis / LLM helpers) that pull in
    # onnxruntime; pymupdf imports them only if present.
    "pymupdf.layout", "pymupdf4llm", "onnxruntime",
    # Not used by the app.
    "tkinter", "matplotlib", "IPython", "pytest", "torchvision", "torchaudio",
    "tensorflow", "flax", "jax", "pandas", "scipy",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtPdf", "PySide6.QtMultimedia",
    "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
]

a = Analysis(
    ["desktop_app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PDFtoWordOCR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can corrupt Qt/PyTorch DLLs
    console=os.environ.get("PDFOCR_CONSOLE") == "1",
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PDFtoWordOCR",
)
