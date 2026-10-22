"""Handwriting recognition with a mocked TrOCR processor/model, and model-loading failures."""

import os
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import FakeModel, FakeProcessor, digital_pdf, handwriting_image, image_pdf
from ocr_app import pipeline
from ocr_app.pipeline import (
    DEFAULT_HANDWRITING_MODEL,
    METHOD_TROCR,
    MODE_AUTO,
    MODE_HANDWRITING,
    HandwritingModelError,
    HandwritingRecognizer,
    process_pdf,
    segment_lines,
)


def test_default_model_is_the_handwritten_checkpoint():
    assert DEFAULT_HANDWRITING_MODEL == "microsoft/trocr-base-handwritten"
    assert HandwritingRecognizer().model_id == "microsoft/trocr-base-handwritten"


# --------------------------------------------------------------------------- segmentation


def test_segment_lines_finds_lines_top_to_bottom():
    image = handwriting_image(["short", "a much longer handwritten line", "medium line"])
    boxes = segment_lines(image)
    assert len(boxes) == 3
    ys = [b[1] for b in boxes]
    assert ys == sorted(ys)
    widths = [b[2] for b in boxes]
    assert widths[0] < widths[2] < widths[1]
    for x, y, w, h in boxes:
        assert 0 <= x and x + w <= image.width and 0 <= y and y + h <= image.height
        assert h < 200  # each box is a single line, not merged neighbours


@pytest.mark.parametrize("fill", [255, 128, 0])
def test_uniform_pages_have_no_lines(fill):
    assert segment_lines(Image.new("RGB", (1200, 1600), (fill, fill, fill))) == []


def test_faint_scanner_noise_has_no_lines():
    rng = np.random.default_rng(0)
    noise = rng.normal(245, 3, size=(1600, 1200)).clip(0, 255).astype(np.uint8)
    assert segment_lines(Image.fromarray(noise).convert("RGB")) == []


# --------------------------------------------------------------------------- recognition (mocked model)


def test_recognize_page_returns_lines_in_reading_order(fake_recognizer):
    image = handwriting_image(["short", "a much longer handwritten line", "medium line"])
    text = fake_recognizer.recognize_page(image)
    assert text == "line 1\nline 2\nline 3"
    crops = fake_recognizer.processor.crops
    assert len(crops) == 3
    assert crops[0].width < crops[2].width < crops[1].width  # crops follow page order
    assert all(c.mode == "RGB" for c in crops)
    assert [len(call) for call in fake_recognizer.processor.calls] == [2, 1]  # batch_size=2


def test_trocr_tokenised_punctuation_is_joined():
    assert pipeline.tidy_trocr_text(" meeting on Friday , at 3 . ") == "meeting on Friday, at 3."
    assert pipeline.tidy_trocr_text("Is it ready ?") == "Is it ready?"
    assert pipeline.tidy_trocr_text("$1,284.50 total") == "$1,284.50 total"
    assert pipeline.tidy_trocr_text('she wrote " see you very soon , "') == (
        'she wrote "see you very soon,"'
    )
    assert pipeline.tidy_trocr_text("2 ) Mr. Smith") == "2) Mr. Smith"
    assert pipeline.tidy_trocr_text("( see note )") == "(see note)"
    assert pipeline.tidy_trocr_text('he said " hello') == 'he said " hello'  # unpaired quote left alone


def test_small_mark_above_a_line_joins_that_line():
    import cv2

    image = handwriting_image(["first line here", "an older friend", "third line"])
    canvas = np.array(image)
    # A short opening-quote mark just above the tall letters of line 2 (which start at y~496).
    cv2.line(canvas, (300, 468), (297, 482), (20, 20, 20), 4)
    cv2.line(canvas, (316, 468), (313, 482), (20, 20, 20), 4)
    boxes = segment_lines(Image.fromarray(canvas))
    assert len(boxes) == 3
    x, y, w, h = boxes[1]
    assert y <= 468 and y + h >= 540  # line 2 now includes the mark


def test_unclear_lines_are_left_out_with_a_warning(monkeypatch, fake_recognizer):
    answers = iter([("Dear Ann,", 0.93), ("World Report of the House", 0.31), ('"', 0.9), ("Thank you", 0.88),
                    ("Dripping candles, 1.000000000000000000", 0.97)])
    monkeypatch.setattr(fake_recognizer, "read_lines", lambda crops: [next(answers) for _ in crops])
    pdf = image_pdf([handwriting_image(["dear ann", "some words", "more words", "thank you", "candles"],
                                       size=(1700, 1800))])
    result = process_pdf(pdf, MODE_HANDWRITING, dpi=150, handwriting_recognizer=fake_recognizer)
    page = result.pages[0]
    # Removed: a guessed line, a punctuation-only line and a confident but looping line.
    assert page.text == "Dear Ann,\nThank you"
    assert page.warnings == ["2 lines were too unclear to read and left out; check the original."]


def test_token_confidence_is_the_mean_probability_ignoring_padding():
    import torch

    class StubModel:
        generation_config = type("Config", (), {"pad_token_id": 1})()

        def compute_transition_scores(self, sequences, scores, normalize_logits):
            assert normalize_logits
            return torch.log(torch.tensor([[0.9, 0.7, 0.5], [0.4, 1.0, 1.0]]))

    recognizer = HandwritingRecognizer(processor=FakeProcessor(), model=StubModel(), device="cpu")
    output = type("Output", (), {"scores": [0, 0, 0]})()
    sequences = torch.tensor([[2, 10, 11, 2], [2, 12, 1, 1]])  # second line is padded
    first, second = recognizer._token_confidences(output, sequences, 2)
    assert first == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert second == pytest.approx(0.4)


def test_auto_mode_only_warns_on_a_low_confidence_scan(monkeypatch, fake_tesseract, fake_recognizer):
    # Auto never switches to the handwriting model (tried and rejected: it misread printed
    # receipts). A page Tesseract reads poorly keeps the Tesseract text plus a warning.
    def fail(*_args, **_kwargs):
        raise AssertionError("Auto mode must not run the handwriting model")

    monkeypatch.setattr(fake_recognizer, "read_lines", fail)
    fake_tesseract.confidence = 47.0
    fake_tesseract.responder = lambda image, i: "Dcar Amn"
    pdf = image_pdf([handwriting_image(["dear ann", "thank you"])])
    result = process_pdf(pdf, MODE_AUTO, tesseract_cmd=fake_tesseract.cmd, handwriting_recognizer=fake_recognizer)
    page = result.pages[0]
    assert page.method == "Tesseract OCR" and page.text == "Dcar Amn"
    assert "Handwriting mode" in page.warnings[0]


def test_generation_uses_the_kv_cache(fake_recognizer):
    fake_recognizer.recognize_page(handwriting_image(["hello"]))
    (options,) = fake_recognizer.model.generate_options
    assert options["use_cache"] is True
    assert options["output_scores"] and options["return_dict_in_generate"]  # for line confidence


def test_blank_page_never_calls_the_model(fake_recognizer):
    assert fake_recognizer.recognize_page(Image.new("RGB", (1000, 1400), "white")) == ""
    assert fake_recognizer.processor.calls == []
    assert fake_recognizer.model.counter == 0


def test_handwriting_mode_multipage_order_and_blank_page(fake_recognizer):
    pdf = image_pdf([
        handwriting_image(["first page"]),
        None,
        handwriting_image(["second", "page two"]),
    ])
    statuses = []
    result = process_pdf(pdf, MODE_HANDWRITING, dpi=150, handwriting_recognizer=fake_recognizer,
                         status_callback=statuses.append)
    assert [p.number for p in result.pages] == [1, 2, 3]
    assert [p.method for p in result.pages] == [METHOD_TROCR] * 3
    assert [p.text for p in result.pages] == ["line 1", "", "line 2\nline 3"]
    assert result.pages[1].warnings == ["No text recognized (blank page?)."]
    assert any("Page 3 of 3: Recognizing handwritten lines" in s for s in statuses)


def test_cpu_is_used_when_cuda_is_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = FakeModel()
    recognizer = HandwritingRecognizer(processor=FakeProcessor(), model=model)
    assert recognizer.recognize_page(handwriting_image(["hello"])) == "line 1"
    assert recognizer.device == "cpu"
    assert model.devices == ["cpu"]
    assert model.generate_devices == ["cpu"]


def test_cuda_is_selected_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pipeline.select_device() == "cuda"
    assert pipeline.select_device("cpu") == "cpu"


def test_falls_back_to_cpu_if_moving_model_to_gpu_fails():
    class GpuBrokenModel(FakeModel):
        def to(self, device):
            if device == "cuda":
                raise RuntimeError("CUDA driver error")
            return super().to(device)

    recognizer = HandwritingRecognizer(processor=FakeProcessor(), model=GpuBrokenModel(), device="cuda")
    recognizer.load()
    assert recognizer.device == "cpu"


# --------------------------------------------------------------------------- lazy loading / missing model


@pytest.fixture
def fake_transformers(monkeypatch):
    """Replace TrOCRProcessor/VisionEncoderDecoderModel.from_pretrained with recorders."""
    import transformers

    calls = []
    behaviour = {"local": "missing", "download": "ok"}

    def make(kind):
        def from_pretrained(source, **kwargs):
            calls.append((kind, source, kwargs))
            phase = "local" if kwargs.get("local_files_only") else "download"
            if behaviour[phase] == "missing":
                raise OSError(f"{source} not found in cache")
            if behaviour[phase] == "network":
                raise ConnectionError("No internet connection")
            return FakeProcessor() if kind == "processor" else FakeModel()

        return from_pretrained

    monkeypatch.setattr(transformers.TrOCRProcessor, "from_pretrained", make("processor"))
    monkeypatch.setattr(transformers.VisionEncoderDecoderModel, "from_pretrained", make("model"))
    return calls, behaviour


def test_model_is_not_loaded_until_needed(fake_transformers):
    calls, _ = fake_transformers
    recognizer = pipeline.get_handwriting_recognizer()
    assert not recognizer.is_loaded
    assert recognizer.recognize_page(Image.new("RGB", (800, 800), "white")) == ""
    assert calls == []  # a blank page does not trigger loading
    assert pipeline.get_handwriting_recognizer() is recognizer  # cached per model


def test_auto_mode_never_loads_the_handwriting_model(monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("handwriting model must not be used in Auto mode")

    monkeypatch.setattr(pipeline, "get_handwriting_recognizer", fail)
    monkeypatch.setattr(HandwritingRecognizer, "load", fail)
    result = process_pdf(digital_pdf(["Digital text that is long enough for Auto mode."]), MODE_AUTO)
    assert result.pages[0].text


def test_cached_model_is_loaded_without_network(fake_transformers, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls, behaviour = fake_transformers
    behaviour["local"] = "ok"
    recognizer = HandwritingRecognizer()
    recognizer.load()
    assert recognizer.is_loaded and recognizer.device == "cpu"
    assert [(kind, kwargs) for kind, _src, kwargs in calls] == [
        ("processor", {"local_files_only": True}),
        ("model", {"local_files_only": True}),
    ]
    recognizer.load()  # second call is a no-op
    assert len(calls) == 2


def test_missing_model_is_downloaded_when_allowed(fake_transformers):
    calls, _ = fake_transformers
    statuses = []
    recognizer = HandwritingRecognizer(allow_download=True)
    recognizer.load(statuses.append)
    assert recognizer.is_loaded
    assert calls[0][2] == {"local_files_only": True}
    assert ("processor", DEFAULT_HANDWRITING_MODEL, {}) in calls
    assert any("Downloading handwriting model" in s for s in statuses)


def test_missing_model_with_download_disabled_raises(fake_transformers):
    calls, _ = fake_transformers
    recognizer = HandwritingRecognizer(allow_download=False)
    with pytest.raises(HandwritingModelError, match="not in the local cache and downloading is disabled"):
        recognizer.load()
    assert all(kwargs == {"local_files_only": True} for _k, _s, kwargs in calls)
    assert not recognizer.is_loaded


def test_failed_download_raises_and_can_be_retried(fake_transformers):
    _, behaviour = fake_transformers
    behaviour["download"] = "network"
    recognizer = HandwritingRecognizer(allow_download=True)
    with pytest.raises(HandwritingModelError, match="Could not download"):
        recognizer.load()
    behaviour["download"] = "ok"
    recognizer.load()
    assert recognizer.is_loaded


def test_missing_model_aborts_handwriting_processing(fake_transformers):
    _, behaviour = fake_transformers
    behaviour["download"] = "network"
    pdf = image_pdf([handwriting_image(["some writing"])])
    with pytest.raises(HandwritingModelError):
        process_pdf(pdf, MODE_HANDWRITING, dpi=100, allow_model_download=True)


def test_missing_transformers_package_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # makes `import transformers` fail
    with pytest.raises(HandwritingModelError, match="PyTorch and Transformers"):
        HandwritingRecognizer().load()


def test_model_folder_next_to_the_app_is_preferred(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "app_base_dir", lambda: tmp_path)
    assert pipeline.resolve_model_source(DEFAULT_HANDWRITING_MODEL) == DEFAULT_HANDWRITING_MODEL
    folder = tmp_path / "models" / "trocr-base-handwritten"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    assert pipeline.resolve_model_source(DEFAULT_HANDWRITING_MODEL) == str(folder)


def test_bad_local_model_folder_is_reported(tmp_path):
    folder = tmp_path / "my-model"
    folder.mkdir()
    with pytest.raises(HandwritingModelError, match="local model folder"):
        HandwritingRecognizer(str(folder)).load()


# --------------------------------------------------------------------------- real model (opt-in)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("PDFOCR_RUN_MODEL_TESTS") != "1",
                    reason="set PDFOCR_RUN_MODEL_TESTS=1 to run the real TrOCR model (needs cached weights)")
def test_real_trocr_model_from_local_cache():
    recognizer = HandwritingRecognizer(allow_download=False)
    try:
        recognizer.load()
    except HandwritingModelError as exc:
        pytest.skip(f"model not cached: {exc}")
    text = recognizer.recognize_page(handwriting_image(["hello world"]))
    assert "hello" in text.lower()
