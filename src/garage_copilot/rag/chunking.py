"""Split markdown documents into heading-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Document:
    path: str
    title: str
    source: str
    body: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_title: str
    source: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_title} > {self.heading}"


def load_document(path: Path) -> Document:
    raw = Path(path).read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    match = _FRONT_MATTER.match(raw)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        raw = raw[match.end() :]
    return Document(
        path=str(path),
        title=meta.get("title", Path(path).stem.replace("-", " ").title()),
        source=meta.get("source", str(path)),
        body=raw,
    )


def chunk_document(doc: Document, max_chars: int = 900) -> list[Chunk]:
    """One chunk per section; long sections are split on paragraph boundaries."""
    sections: list[tuple[str, list[str]]] = []
    heading = doc.title
    lines: list[str] = []
    for line in doc.body.splitlines():
        match = _HEADING.match(line)
        if match:
            if lines:
                sections.append((heading, lines))
            heading, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    if lines:
        sections.append((heading, lines))

    chunks: list[Chunk] = []
    for heading, body_lines in sections:
        paragraphs = [p.strip() for p in "\n".join(body_lines).split("\n\n") if p.strip()]
        for part_no, text in enumerate(_pack(paragraphs, max_chars)):
            chunk_id = f"{Path(doc.path).stem}#{_slug(heading)}-{part_no}"
            chunks.append(Chunk(chunk_id, doc.title, doc.source, heading, text))
    return chunks


def load_directory(
    directory: Path, max_chars: int = 900, only: tuple[str, ...] | None = None
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        if only is not None and path.stem not in only:
            continue
        chunks.extend(chunk_document(load_document(path), max_chars=max_chars))
    return chunks


def _pack(paragraphs: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            packed.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        packed.append(current)
    return packed


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
