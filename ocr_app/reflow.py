"""Join hard-wrapped lines back into paragraphs and repair words split across lines.

PDF text layers and OCR output contain one entry per printed line. Exported
one-to-one, every line becomes its own Word paragraph and editing a sentence no
longer reflows. These helpers rebuild paragraphs:

* ``page_paragraphs`` uses PyMuPDF's line geometry (indents, whether the next word
  would have fitted on the previous line, centred lines, column breaks) for PDFs
  with a text layer.
* ``reflow_text`` works on plain OCR text, using blank lines and line lengths.

Hyphenated line ends ("intel-" + "ligent") are joined using the document's own
vocabulary: the hyphen is kept when the hyphenated form appears elsewhere in the
text ("instruction-tuning") and dropped when the joined word does
("intelligent"). With no evidence either way it is dropped before a lowercase
continuation, which is what typesetting hyphenation usually is.
"""

from __future__ import annotations

import re
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]")
_HEAD_RE = re.compile(r"([A-Za-z][A-Za-z\-]*)-$")
_TAIL_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")
_SENTENCE_END = (".", "!", "?", ":", ";", '"', "”", ")")
_LIST_MARKER_RE = re.compile(r"^(\[\d+\]|\(?\d{1,3}[.)]\s|[•●▪*–—-]\s)")


def build_vocabulary(texts: Iterable[str]) -> set[str]:
    """Lower-cased words (including hyphenated compounds) that occur in the texts,
    ignoring line-end hyphenation fragments."""
    vocabulary: set[str] = set()
    for text in texts:
        for line in text.splitlines():
            tokens = line.split()
            for index, token in enumerate(tokens):
                if index == len(tokens) - 1 and token.endswith("-"):
                    continue
                for word in _WORD_RE.findall(token):
                    word = word.lower()
                    vocabulary.add(word)
                    if "-" in word:  # "few-shot" also teaches "few" and "shot"
                        vocabulary.update(part for part in word.split("-") if part)
    return vocabulary


def join_lines(left: str, right: str, vocabulary: set[str]) -> str:
    """Join two consecutive lines of one paragraph, repairing a hyphenated word."""
    left, right = left.rstrip(), right.lstrip()
    if not left or not right:
        return left or right
    head = _HEAD_RE.search(left)
    tail = _TAIL_RE.match(right)
    if head and tail and not left.endswith("--"):
        head_word, tail_word = head.group(1), tail.group(0)
        hyphenated = f"{head_word}-{tail_word}".lower()
        joined = f"{head_word}{tail_word}".lower()
        if hyphenated in vocabulary and joined not in vocabulary:
            keep_hyphen = True
        elif joined in vocabulary:
            keep_hyphen = False
        else:
            # No direct evidence: keep the hyphen inside an already hyphenated compound
            # ("Chain-of-" + "restoration"), between two known words ("above-mentioned")
            # and before a capital ("RL-Restore"); drop it for syllable breaks.
            last_part = head_word.rsplit("-", 1)[-1].lower()
            both_words = (
                len(last_part) >= 3 and len(tail_word) >= 3
                and last_part in vocabulary and tail_word.lower() in vocabulary
            )
            keep_hyphen = "-" in head_word or both_words or tail_word[0].isupper()
        return left + right if keep_hyphen else left[:-1] + right
    return f"{left} {right}"


def join_paragraph(lines: list[str], vocabulary: set[str]) -> str:
    text = ""
    for line in lines:
        text = join_lines(text, line, vocabulary)
    return text


def reflow_text(text: str, vocabulary: set[str]) -> str:
    """Reflow plain text (e.g. Tesseract output): blank lines separate paragraphs; inside a
    block, a line ends its paragraph when it is very short, or short and ends a sentence."""
    paragraphs: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        longest = max(len(line) for line in lines)
        current: list[str] = []
        for line in lines:
            current.append(line)
            short = len(line) < 0.6 * longest
            ends_sentence = line.endswith(_SENTENCE_END) and len(line) < 0.9 * longest
            if short or ends_sentence:
                paragraphs.append(join_paragraph(current, vocabulary))
                current = []
        if current:
            paragraphs.append(join_paragraph(current, vocabulary))
    return "\n\n".join(paragraphs)


def _visual_lines(block: dict) -> list[dict]:
    """Merge PyMuPDF lines that share a baseline (it sometimes splits one printed line)."""
    pieces = []
    for line in block.get("lines", []):
        text = "".join(span["text"] for span in line["spans"]).strip()
        if not text:
            continue
        x0, y0, x1, y1 = line["bbox"]
        horizontal = abs(line.get("dir", (1, 0))[1]) < 0.1
        pieces.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text, "horizontal": horizontal})
    pieces.sort(key=lambda p: (p["y0"], p["x0"]))
    lines: list[dict] = []
    for piece in pieces:
        if lines and piece["horizontal"] and lines[-1]["horizontal"]:
            last = lines[-1]
            overlap = min(last["y1"], piece["y1"]) - max(last["y0"], piece["y0"])
            if overlap > 0.5 * min(last["y1"] - last["y0"], piece["y1"] - piece["y0"]):
                last["parts"].append(piece)
                last["x0"], last["x1"] = min(last["x0"], piece["x0"]), max(last["x1"], piece["x1"])
                last["y0"], last["y1"] = min(last["y0"], piece["y0"]), max(last["y1"], piece["y1"])
                continue
        lines.append({**piece, "parts": [piece]})
    for line in lines:
        line["text"] = " ".join(p["text"] for p in sorted(line["parts"], key=lambda p: p["x0"]))
    return lines


def _page_lines(page, flags: int) -> list[dict]:
    """All visual lines of a page in reading order, each annotated with the margins of
    its PyMuPDF block. Blocks do not always coincide with paragraphs (a block may end
    with the first line of the next paragraph), so paragraphs are built across blocks."""
    lines: list[dict] = []
    for block in page.get_text("dict", flags=flags)["blocks"]:
        if block.get("type") != 0:
            continue
        block_lines = _visual_lines(block)
        if not block_lines:
            continue
        # The x position continuation lines start at. Normal paragraphs indent their
        # first line and hanging indents (reference lists) outdent it; either way a line
        # deviating from it starts a paragraph. Lines opening with a list marker ("[3]",
        # "2.", bullets) are paragraph starts, so not counted; nor is the block's first
        # line (often an indented paragraph start) unless nothing else is left.
        unmarked = [line for line in block_lines if not _LIST_MARKER_RE.match(line["text"])]
        candidates = unmarked[1:] if unmarked and unmarked[0] is block_lines[0] else unmarked
        starts = [round(line["x0"]) for line in (candidates or unmarked or block_lines)]
        body_x = max(sorted(set(starts)), key=starts.count)
        for line in block_lines:
            line["body_x"] = body_x
            lines.append(line)
    _add_column_right_edges(lines, page.rect.width)
    return lines


def _add_column_right_edges(lines: list[dict], page_width: float) -> None:
    """Estimate the right edge of the text column each line was wrapped in.

    Lines are grouped into columns by their block's continuation x (indents do not
    change it); the edge is the 80th percentile of the column's line ends, so a few
    full-width lines (titles, footers) do not distort it. A narrow run of short lines
    with no text to its right that does not look wrapped (e.g. an address on an
    otherwise empty page) extends to the mirrored page margin, so its lines are not
    mistaken for full-width text.
    """
    if not lines:
        return
    page_left = min(line["x0"] for line in lines)
    text_width = max(1.0, page_width - 2 * page_left)
    columns: list[list[dict]] = []
    for line in sorted(lines, key=lambda item: item["body_x"]):
        if columns and abs(line["body_x"] - columns[-1][0]["body_x"]) <= 0.03 * page_width:
            columns[-1].append(line)
        else:
            columns.append([line])
    for column in columns:
        ends = sorted(line["x1"] for line in column)
        right = ends[min(len(ends) - 1, int(0.8 * len(ends)))]
        left = min(line["x0"] for line in column)
        # Wrapped text has several lines ending near the column edge; a list of short
        # lines (an address) has at most one, so its edge is not a wrapping boundary.
        reaching_edge = sum(1 for line in column if line["x1"] >= right - 0.1 * (right - left))
        nothing_to_the_right = not any(other["x0"] >= right for other in lines)
        if nothing_to_the_right and right - left < 0.5 * text_width and reaching_edge <= 1:
            right = max(right, page_width - page_left)
        for line in column:
            line["column_right"] = right


def _next_word_would_fit(previous: dict, line: dict) -> bool:
    """True if the line's first word would have fitted at the end of the previous line,
    i.e. the line break was deliberate rather than word wrapping."""
    text = line["text"]
    char_width = (line["x1"] - line["x0"]) / max(1, len(text))
    first_word = text.split()[0] if text.split() else text
    needed = (len(first_word) + 1) * char_width
    return previous["x1"] + needed < previous["column_right"] - 0.5 * char_width


def _starts_paragraph(previous: dict, line: dict, current_text: str) -> bool:
    height = max(1.0, line["y1"] - line["y0"])
    if not (line["horizontal"] and previous["horizontal"]):
        return True
    gap = line["y0"] - previous["y1"]
    directly_below = (
        -0.5 * height <= gap <= 1.0 * height
        and min(previous["x1"], line["x1"]) > max(previous["x0"], line["x0"])
    )
    if not directly_below:  # new column, new region or a large gap
        return not _continues(current_text, line["text"])
    if abs(line["x0"] - line["body_x"]) > 0.8 * height:  # indented or outdented start
        return True
    if _centred_pair(previous, line):  # titles, author lists: one entry per line
        return True
    if re.search(r"[A-Za-z]-$", previous["text"]):  # word split across the lines
        return False
    return _next_word_would_fit(previous, line)


def _centred_pair(previous: dict, line: dict) -> bool:
    """Both edges moved by the same amount in opposite directions: centred lines.
    (Justified or left-aligned text keeps its left edge; an indent moves only one edge.)"""
    dx0, dx1 = line["x0"] - previous["x0"], line["x1"] - previous["x1"]
    return abs(dx0) > 2 and abs(dx1) > 2 and (dx0 > 0) != (dx1 > 0) and abs(dx0 + dx1) < 4


def _continues(previous: str, following: str) -> bool:
    """Text cut at a column or page-region break that clearly goes on: a word split with a
    hyphen, or a long paragraph stopped mid-sentence and resumed in lower case.
    Short fragments (figure labels) are not joined."""
    if re.search(r"[A-Za-z]-$", previous) and following[:1].isalpha():
        return True
    return len(previous) >= 60 and not previous.endswith(_SENTENCE_END) and following[:1].islower()


def page_paragraphs(page, vocabulary: set[str], flags: int) -> str:
    """Paragraph text of a PDF page's text layer, in PyMuPDF's reading order."""
    paragraphs: list[str] = []
    current = ""
    previous = None
    for line in _page_lines(page, flags):
        if previous is not None and _starts_paragraph(previous, line, current):
            paragraphs.append(current)
            current = ""
        current = join_lines(current, line["text"], vocabulary)
        previous = line
    if current:
        paragraphs.append(current)
    return "\n\n".join(p for p in paragraphs if p)
