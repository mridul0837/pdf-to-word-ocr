"""Joining printed lines into paragraphs and repairing hyphenated words."""

import pymupdf
import pytest

from conftest import digital_pdf, image_pdf, text_image
from ocr_app.pipeline import DICT_FLAGS, MODE_AUTO, MODE_PRINTED, process_pdf
from ocr_app.reflow import build_vocabulary, join_lines, page_paragraphs, reflow_text


@pytest.mark.parametrize(
    ("left", "right", "vocab_text", "expected"),
    [
        ("the system is intel-", "ligent and fast", "an intelligent agent", "the system is intelligent and fast"),
        ("we build an instruction-", "tuning dataset", "the instruction-tuning set", "we build an instruction-tuning dataset"),
        ("as noted in the above-", "mentioned work", "it is above what was mentioned", "as noted in the above-mentioned work"),
        ("models such as Chain-of-", "restoration work", "", "models such as Chain-of-restoration work"),
        ("as shown by RL-", "Restore results", "", "as shown by RL-Restore results"),
        ("a convolu-", "tional network", "", "a convolutional network"),  # no evidence: syllable break
        ("first line", "second line", "", "first line second line"),
        ("pages 12--", "15 of the book", "", "pages 12-- 15 of the book"),
    ],
)
def test_join_lines_repairs_hyphenation(left, right, vocab_text, expected):
    assert join_lines(left, right, build_vocabulary([vocab_text])) == expected


def test_vocabulary_ignores_line_end_fragments_and_learns_compound_parts():
    vocab = build_vocabulary(["a few-shot learner is intel-\nligent"])
    assert {"few-shot", "few", "shot", "learner"} <= vocab
    assert "intel" not in vocab


def test_reflow_text_uses_blank_lines_and_short_lines():
    text = (
        "This is the first line of a long paragraph that\n"
        "continues on the next line and ends here.\n"
        "\n"
        "Heading\n"
        "Another paragraph starts after the heading and it\n"
        "runs over two lines as well, with a split exam-\n"
        "ple word."
    )
    paragraphs = reflow_text(text, build_vocabulary([text, "for example"])).split("\n\n")
    assert paragraphs == [
        "This is the first line of a long paragraph that continues on the next line and ends here.",
        "Heading",
        "Another paragraph starts after the heading and it runs over two lines as well, with a split example word.",
    ]


def _two_column_page():
    """A page laid out like a paper: indented paragraph starts, short last lines, a word split
    across the column break, and a reference list with hanging indents."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    left = [
        (70, "The first paragraph starts with an indent and"),
        (58, "continues over several lines of the left col-"),
        (58, "umn until it finally ends right here."),
        (70, "A second paragraph also starts with an indent,"),
        (58, "and its text runs to the bottom of the column with"),
        (58, "a word that is hyphen-"),
    ]
    right = [
        (318, "ated across the column break and then ends."),
        (318, "[1] Ada Lovelace and Charles Babbage. Notes on the"),
        (330, "analytical engine. Scientific Memoirs, 1843."),
        (318, "[2] Alan Turing. On computable numbers, with an"),
        (330, "application to the Entscheidungsproblem, 1936."),
    ]
    for column in (left, right):
        y = 100
        for x, text in column:
            page.insert_text((x, y), text, fontsize=9)
            y += 12
    return doc


def test_page_paragraphs_rebuilds_paper_layout():
    doc = _two_column_page()
    page = doc[0]
    vocab = build_vocabulary([page.get_text()])
    paragraphs = page_paragraphs(page, vocab, DICT_FLAGS).split("\n\n")
    assert paragraphs == [
        "The first paragraph starts with an indent and continues over several lines of the left column "
        "until it finally ends right here.",
        "A second paragraph also starts with an indent, and its text runs to the bottom of the column with "
        "a word that is hyphenated across the column break and then ends.",
        "[1] Ada Lovelace and Charles Babbage. Notes on the analytical engine. Scientific Memoirs, 1843.",
        "[2] Alan Turing. On computable numbers, with an application to the Entscheidungsproblem, 1936.",
    ]


def test_join_lines_option_controls_embedded_text():
    data = _two_column_page().tobytes()
    joined = process_pdf(data, MODE_AUTO).pages[0].text
    kept = process_pdf(data, MODE_AUTO, join_lines=False).pages[0].text
    assert "left column until" in joined and "\n" not in joined.split("\n\n")[0]
    assert "left col-\numn" in kept  # original line breaks preserved when switched off


def test_receipt_lines_are_kept_but_wrapped_sentences_joined():
    receipt = (
        "CORNER CAFE\n12 MAIN STREET\nServer: Alex\nTotal $18.40\nTip $3.00\n"
        "Please keep this copy for your\nrecords\nTable: T12"
    )
    assert reflow_text(receipt, set()).split("\n") == [
        "CORNER CAFE", "12 MAIN STREET", "Server: Alex", "Total $18.40", "Tip $3.00",
        "Please keep this copy for your records", "Table: T12",
    ]


def test_prose_with_long_lines_is_not_a_list():
    from ocr_app.reflow import looks_like_a_list

    prose = ["The design goals required a small, lightweight, expendable unit, suitable",
             "for use with smaller sounding rockets and compatible with existing ground"]
    assert not looks_like_a_list(prose)
    assert looks_like_a_list(["Subtotal $12.87", "Tax $0.90", "Total $13.77"])
    assert looks_like_a_list(["Item one ......... 12.50", "Another longer item name ...... 3.25", "x 10"])


def test_tesseract_output_is_joined(fake_tesseract):
    fake_tesseract.responder = lambda image, i: "A scanned paragraph that wraps\nonto a second line.\n\nNext one.\n\x0c"
    result = process_pdf(image_pdf([text_image("x")]), MODE_PRINTED, tesseract_cmd=fake_tesseract.cmd)
    assert result.pages[0].text == "A scanned paragraph that wraps onto a second line.\n\nNext one."


def test_handwriting_keeps_its_lines(fake_recognizer):
    from conftest import handwriting_image

    pdf = image_pdf([handwriting_image(["first line", "second line"])])
    result = process_pdf(pdf, "Handwriting", dpi=150, handwriting_recognizer=fake_recognizer, join_lines=True)
    assert result.pages[0].text == "line 1\nline 2"


def test_centred_title_block_lines_stay_separate():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for y, text in [(100, "Jane Researcher John Scientist"), (114, "University of Somewhere"), (128, "{jane, john}@example.edu")]:
        width = pymupdf.get_text_length(text, fontsize=10)
        page.insert_text((306 - width / 2, y), text, fontsize=10)
    page.insert_text((470, 300), "a label on the right", fontsize=8)  # text to the right of the block
    paragraphs = page_paragraphs(page, set(), DICT_FLAGS).split("\n\n")
    assert paragraphs[:3] == ["Jane Researcher John Scientist", "University of Somewhere", "{jane, john}@example.edu"]


def test_text_wrapped_in_a_narrow_column_is_joined():
    paragraph = (
        "All documents stay on this computer: nothing is uploaded, no account is needed and the "
        "app works offline. Digital PDFs are read from their text layer."
    )
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 110, 330, 400), paragraph, fontsize=10)
    assert len(page.get_text().splitlines()) == 3  # wrapped by width
    assert page_paragraphs(page, set(), DICT_FLAGS) == paragraph


def test_short_lines_like_addresses_stay_separate():
    text = "Jane Doe\n12 High Street\nSpringfield\n\nDear John,"
    result = process_pdf(digital_pdf([text]))
    assert result.pages[0].text.split("\n\n") == ["Jane Doe", "12 High Street", "Springfield", "Dear John,"]
