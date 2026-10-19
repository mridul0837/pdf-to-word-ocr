# PDF to Word OCR — Windows desktop app

A native Windows desktop application (Python + PySide6) that turns PDFs into editable
Microsoft Word (`.docx`) files. All processing is local: no cloud OCR, no paid APIs.

- **Digital PDFs**: the embedded text layer is extracted directly (PyMuPDF).
- **Scanned printed documents**: OCR with [Tesseract](https://github.com/tesseract-ocr/tesseract).
- **Handwritten documents**: line-by-line recognition with the pretrained
  [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) model.

The recognized text appears in an editable preview so you can correct OCR mistakes
before saving the Word document.

## Recognition modes

| Mode | What it does | Needs |
|---|---|---|
| **Auto** (default) | Uses each page's embedded text when it has some; pages without usable text (scans) are OCR'd with Tesseract. | Tesseract only for scanned pages |
| **Printed Text** | Renders every page and OCRs it with Tesseract, ignoring any embedded text. | Tesseract |
| **Handwriting** | Renders every page, finds text lines with OpenCV and reads each line with TrOCR. | TrOCR model (downloaded once, ~1.3 GB) |

**Auto mode does not detect handwriting.** A handwritten scan in Auto mode is sent to
Tesseract, which handles handwriting poorly. Choose **Handwriting** yourself for handwritten pages.
A PDF that mixes printed and handwritten pages has to be processed twice (once per mode),
or the pages split up.

In Auto mode, if Tesseract is missing, digital pages are still extracted and the scanned
pages are listed in a warning. If no page yields any text, the run fails with the
"Tesseract not found" message.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.11–3.13 (developed and tested with 3.13.9)
- Tesseract OCR 5.x with language data, for Auto/Printed modes on scanned pages
- About 1.3 GB of disk space and a one-time internet connection for the handwriting model
- Optional: an NVIDIA GPU with a CUDA build of PyTorch, to speed up handwriting.
  The app falls back to the CPU automatically when CUDA is unavailable.

## 1. Set up the Python environment (PowerShell)

```powershell
cd "D:\Z. OCR App\pdf_to_word_desktop_app"     # the project folder
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # app + pytest + PyInstaller
```

`requirements.txt` holds only the runtime dependencies. `requirements-dev.txt` adds pytest and PyInstaller.

The examples call `.\.venv\Scripts\python.exe` directly, so the venv never needs activating.
If you prefer to activate it (`.\.venv\Scripts\Activate.ps1`) and PowerShell blocks the
script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**GPU (optional):** on Windows, `pip install torch` installs a CPU-only build. For CUDA,
first install the CUDA build of PyTorch with the command from
<https://pytorch.org/get-started/locally/> (Windows / Pip / CUDA), then install the requirements.
A CUDA build makes the packaged app several GB larger.

## 2. Install and configure Tesseract

1. Install the UB Mannheim build of Tesseract, either from
   <https://github.com/UB-Mannheim/tesseract/wiki> or with
   `winget install --id UB-Mannheim.TesseractOCR`.
   Tick any extra languages you need during installation (English is included).
2. Check it works:
   ```powershell
   & "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   & "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
   ```

The app finds `tesseract.exe` automatically, checking these locations in order:

1. The path entered in the app's **Settings → Tesseract executable** (use **Test** to check it).
2. The `TESSERACT_CMD` environment variable.
3. A `tesseract\` folder next to the app (or next to `PDFtoWordOCR.exe`).
4. `PATH`.
5. The standard install folders: `%ProgramFiles%\Tesseract-OCR`, `%ProgramFiles(x86)%\Tesseract-OCR`
   and `%LOCALAPPDATA%\Programs\Tesseract-OCR`.

If Tesseract reports no languages, its language data is not where it looks. Put the
`*.traineddata` files in its `tessdata` folder, or set `TESSDATA_PREFIX` to the folder
that contains them. The **Language** box accepts Tesseract codes such as `eng`, `fra`
or `eng+fra`, and each one must be installed.

## 3. Handwriting model: first download and caching

The model loads only when Handwriting mode is used. The app looks in the local cache
first, without contacting the network. Only if the weights are missing and **Settings →
Allow downloading the handwriting model** is ticked does it download them, about 1.3 GB,
once. They are cached in `%USERPROFILE%\.cache\huggingface\hub` (set `HF_HOME` to move
the cache). The development setup and the packaged `.exe` share that cache, and later
launches reuse it offline.

To download ahead of time:

```powershell
.\.venv\Scripts\python.exe -m ocr_app.fetch_model
```

For a PC without internet, save a self-contained copy and put it next to the app:

```powershell
.\.venv\Scripts\python.exe -m ocr_app.fetch_model --output models\trocr-base-handwritten
# then copy the "models" folder next to desktop_app.py or PDFtoWordOCR.exe
```

The app prefers `models\trocr-base-handwritten` beside it when that folder exists. The
**Handwriting model** setting also accepts another Hugging Face TrOCR model id or a local folder.

The model recognizes **English** handwriting only, and the Language box does not apply to it.

## 4. Run the app

```powershell
.\.venv\Scripts\python.exe desktop_app.py            # or: desktop_app.py "C:\path\file.pdf"
```

1. **Choose PDF…** (or drag a PDF onto the window). The output path defaults to `<name>_ocr.docx` beside the PDF.
2. Pick a **Mode**, plus the **Language** and **DPI** (250–300 suits most scans).
3. Click **Extract text**. Processing runs in a background thread, so the window stays
   responsive. The progress bar counts pages, the status line shows the current step,
   and **Cancel** stops after the current step.
4. Review and edit the text. Lines like `===== PAGE 2 · Tesseract OCR =====` mark page
   boundaries and become page breaks in Word. Delete them if you don't want page breaks.
5. Click **Save Word document**.

Settings (mode, language, DPI, Tesseract path, model and download option) are remembered
in the registry under `HKEY_CURRENT_USER\Software\PDFtoWordOCR`.

## 5. Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite needs no internet, model weights, GPU or Tesseract. External engines are
mocked, and the GUI tests run on Qt's offscreen platform. Two opt-in integration tests
use the real engines and skip themselves otherwise:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"   # real Tesseract test
$env:PDFOCR_RUN_MODEL_TESTS = "1"                                     # real TrOCR test (cached model)
.\.venv\Scripts\python.exe -m pytest -m integration
```

## 6. Build the Windows executable

```powershell
.\build_exe.ps1                 # tests -> PyInstaller (one-folder) -> self-test of the .exe
.\build_exe.ps1 -SkipTests      # skip the pytest run
.\build_exe.ps1 -Console        # console build, to see tracebacks while debugging
```

If script execution is blocked, run `powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`.

Output: `dist\PDFtoWordOCR\PDFtoWordOCR.exe`. The build uses `PDFtoWordOCR.spec` in
one-folder mode, so ship or copy the **whole** `dist\PDFtoWordOCR` folder, not just the `.exe`.
The spec bundles Qt's platform plugins, PyTorch and the transformers modules TrOCR needs.
It leaves out PyMuPDF's optional layout/onnxruntime add-ons and unused Qt modules.

To check a build on any PC:

```powershell
.\dist\PDFtoWordOCR\PDFtoWordOCR.exe --self-test report.json
```

This writes a JSON report covering imports, digital-PDF extraction, DOCX export and GUI
start-up, plus Tesseract and the cached handwriting model when available. It never downloads anything.

**Not bundled:** these must exist on the target PC.

- **Tesseract**: install it there, or copy a Tesseract folder (with `tessdata`) to
  `dist\PDFtoWordOCR\tesseract\`.
- **TrOCR weights**: downloaded on first handwriting use, or copied as
  `dist\PDFtoWordOCR\models\trocr-base-handwritten` (see section 3).

Check the licences of Tesseract (Apache-2.0), the TrOCR model and the bundled libraries
before redistributing the app.

## Project layout

```
desktop_app.py            PySide6 GUI and entry point (also --self-test)
ocr_app/pipeline.py       PDF opening, page rendering, Tesseract, line segmentation, TrOCR, orchestration
ocr_app/docx_export.py    Edited text -> .docx (page markers -> page breaks)
ocr_app/utils.py          Text cleanup, page markers, app/user paths
ocr_app/fetch_model.py    Pre-download / save the handwriting model
ocr_app/selftest.py       Headless checks used to verify a build
tests/                    pytest suite
PDFtoWordOCR.spec         PyInstaller spec
build_exe.ps1             Reproducible build script
```

## Limitations

- Layout is not preserved. The output is plain paragraphs (one per text line) with a page
  break between pages. Columns, tables, images, fonts, headers and equations are not reconstructed.
- Auto mode trusts any page with at least 20 characters of embedded text. A page that
  mixes a text layer with scanned images uses only the text layer; use Printed Text mode for such files.
- Handwriting line detection is a simple OpenCV heuristic. It works best on clearly
  separated horizontal lines. Side-by-side columns get merged into one line, and dense
  cursive, skewed or overlapping lines and margin notes may be split or read poorly.
  TrOCR sometimes adds a full stop at the end of a line.
- Handwriting recognition is slow on CPU: a few seconds per line, plus a one-time model
  load of about 10–20 s. A GPU helps substantially.
- Password-protected PDFs are rejected. Remove the password first.
- OCR quality depends on scan quality. Proofread names, numbers, dates and technical content.

## Troubleshooting

- **"Tesseract OCR was not found"**: install Tesseract (section 2), or set its path in
  Settings and click **Test**.
- **"found no language data" / "language data not installed"**: add the `.traineddata`
  files to Tesseract's `tessdata` folder, or set `TESSDATA_PREFIX`.
- **Handwriting model errors**: tick *Allow downloading…* and connect to the internet once,
  or run `python -m ocr_app.fetch_model`. Behind a proxy, set `HTTPS_PROXY`.
- **The packaged app closes or misbehaves**: run `PDFtoWordOCR.exe --self-test report.json`,
  build a console version with `.\build_exe.ps1 -Console`, and check
  `%LOCALAPPDATA%\PDFtoWordOCR\app.log`, where library output from the windowed app is written.
- **"Could not write … open in Word"**: close the document in Word and save again.
