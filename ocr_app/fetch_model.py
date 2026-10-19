"""Download the handwriting model once so the app can run offline.

Usage (PowerShell, inside the project's virtual environment):

    python -m ocr_app.fetch_model                      # into the Hugging Face cache
    python -m ocr_app.fetch_model --output models\\trocr-base-handwritten

Weights in the Hugging Face cache (%USERPROFILE%\\.cache\\huggingface\\hub) are
reused by both ``python desktop_app.py`` and the packaged .exe. A folder
written with ``--output`` can be copied next to PDFtoWordOCR.exe as
``models\\trocr-base-handwritten`` for machines without internet access.
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import DEFAULT_HANDWRITING_MODEL, HandwritingModelError, HandwritingRecognizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_HANDWRITING_MODEL, help="Hugging Face model id")
    parser.add_argument("--output", help="also save a self-contained copy to this folder")
    args = parser.parse_args(argv)

    recognizer = HandwritingRecognizer(args.model, allow_download=True, device="cpu")
    try:
        recognizer.load(status_callback=print)
    except HandwritingModelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Model '{args.model}' is available in the local cache.")
    if args.output:
        recognizer.processor.save_pretrained(args.output)
        recognizer.model.save_pretrained(args.output)
        print(f"Saved a copy to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
