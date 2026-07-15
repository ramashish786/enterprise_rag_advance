"""Structure-aware chunker with parent-child hierarchy.

Blocks between two headings form one "parent chunk": a large, coherent
unit for context. Children are ~300-token sliding-window slices of that
parent's text. Retrieval matches against children (precise), but the
parent gets sent to the LLM (coherent).

Tokens are approximated as ~4 characters, the usual rule of thumb for
English. Precise counting (tiktoken) lives in the production embedder
instead, so this module stays dependency-light for tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
from uuid import UUID, uuid4

from app.rag.parser import ParsedDocument, TextBlock

CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    id: UUID
    parent_id: UUID | None
    index: int
    content: str
    approx_tokens: int
    heading_path: list[str] = field(default_factory=list)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_document(
    doc: ParsedDocument,
    *,
    child_target_tokens: int = 300,
    child_overlap_tokens: int = 40,
    parent_max_tokens: int = 1500,
) -> tuple[list[Chunk], list[Chunk]]:
    """Return (parents, children).

    Each parent covers a coherent section. Children are sliding-window slices
    of the parent's text with `parent_id` set.
    """
    parents: list[Chunk] = []
    children: list[Chunk] = []

    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current_blocks: list[TextBlock] = []
    current_heading_path: list[str] = []
    parent_index = 0

    def flush_parent() -> None:
        nonlocal parent_index
        if not current_blocks:
            return
        text = "\n".join(b.text for b in current_blocks).strip()
        if not text:
            return
        # If the parent is too large, split into multiple parent-sized units.
        for piece in _sliding_windows(text, parent_max_tokens, overlap_tokens=0):
            parent = Chunk(
                id=uuid4(),
                parent_id=None,
                index=parent_index,
                content=piece,
                approx_tokens=_approx_tokens(piece),
                heading_path=list(current_heading_path),
            )
            parents.append(parent)
            child_index = 0
            for cpiece in _sliding_windows(piece, child_target_tokens, child_overlap_tokens):
                children.append(Chunk(
                    id=uuid4(),
                    parent_id=parent.id,
                    index=child_index,
                    content=cpiece,
                    approx_tokens=_approx_tokens(cpiece),
                    heading_path=list(current_heading_path),
                ))
                child_index += 1
            parent_index += 1

    for block in doc.blocks:
        if block.kind == "heading":
            # flush current section
            flush_parent()
            current_blocks = []
            # update heading path
            while heading_stack and heading_stack[-1][0] >= block.heading_level:
                heading_stack.pop()
            heading_stack.append((block.heading_level, block.text))
            current_heading_path = [h[1] for h in heading_stack]
        else:
            current_blocks.append(block)

    flush_parent()
    return parents, children


def _sliding_windows(text: str, target_tokens: int, overlap_tokens: int) -> Iterable[str]:
    """Yield windows of approximately `target_tokens` with `overlap_tokens` overlap.

    Splits on paragraph, then on sentence, then on word to keep windows
    coherent.
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    if len(text) <= target_chars:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        window = text[start:end]

        # Try to end at a sentence boundary in the last 20% of the window.
        if end < len(text):
            tail_start = int(len(window) * 0.8)
            tail = window[tail_start:]
            for sep in [". ", "! ", "? ", "\n"]:
                idx = tail.rfind(sep)
                if idx >= 0:
                    window = window[: tail_start + idx + 1]
                    end = start + len(window)
                    break

        yield window.strip()
        if end >= len(text):
            break
        start = end - overlap_chars
        if start < 0:
            start = 0
