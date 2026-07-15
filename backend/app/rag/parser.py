"""Document parsers.

Each parser returns a `ParsedDocument` with a list of `TextBlock`. Blocks
carry heading level so downstream chunkers can preserve section boundaries.

We use `unstructured` where available (best layout awareness), otherwise fall
back to per-format libraries. Both paths return the same shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class TextBlock:
    text: str
    kind: str  # heading / paragraph / list_item / table / caption
    heading_level: int = 0  # 0 for non-headings


@dataclass
class ParsedDocument:
    filename: str
    blocks: list[TextBlock] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


# --- Format-specific parsers ---

def parse_pdf(path: str | Path) -> ParsedDocument:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    blocks: list[TextBlock] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for para in _split_paragraphs(text):
            blocks.append(TextBlock(text=para, kind="paragraph"))
    return ParsedDocument(filename=Path(path).name, blocks=blocks, metadata={"pages": len(reader.pages)})


def parse_docx(path: str | Path) -> ParsedDocument:
    from docx import Document as Doc
    doc = Doc(str(path))
    blocks: list[TextBlock] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            try:
                level = int(style.split()[-1])
            except (ValueError, IndexError):
                level = 1
            blocks.append(TextBlock(text=text, kind="heading", heading_level=level))
        else:
            blocks.append(TextBlock(text=text, kind="paragraph"))
    # tables → flatten each row into a text block
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                blocks.append(TextBlock(text=row_text, kind="table"))
    return ParsedDocument(filename=Path(path).name, blocks=blocks)


def parse_pptx(path: str | Path) -> ParsedDocument:
    from pptx import Presentation
    pres = Presentation(str(path))
    blocks: list[TextBlock] = []
    for slide_num, slide in enumerate(pres.slides, start=1):
        for shape in slide.shapes:
            if not hasattr(shape, "text_frame"):
                continue
            for para in shape.text_frame.paragraphs:
                text = " ".join(run.text for run in para.runs).strip()
                if text:
                    blocks.append(TextBlock(text=text, kind="paragraph"))
    return ParsedDocument(filename=Path(path).name, blocks=blocks, metadata={"slides": len(pres.slides)})


def parse_plain_text(path: str | Path) -> ParsedDocument:
    """Fallback for txt/md, and useful for tests."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = [TextBlock(text=p, kind="paragraph") for p in _split_paragraphs(text)]
    return ParsedDocument(filename=Path(path).name, blocks=blocks)


# --- Router ---

_PARSERS = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".txt":  parse_plain_text,
    ".md":   parse_plain_text,
}


def parse(path: str | Path) -> ParsedDocument:
    ext = Path(path).suffix.lower()
    parser = _PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported document format: {ext}")
    return parser(path)


# --- Helpers ---

def _split_paragraphs(text: str) -> Iterable[str]:
    for chunk in text.split("\n\n"):
        stripped = chunk.strip()
        if stripped:
            yield stripped
