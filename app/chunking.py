"""Recursive character text splitting for KB ingestion.

Markdown documents are first split into heading-scoped sections; each
section is then split recursively by trying progressively finer separators
(paragraph breaks, then lines, then sentences, then words) until every piece
fits within `chunk_size` characters. Adjacent pieces are merged back together
up to `chunk_size` with `chunk_overlap` characters of trailing context carried
into the next chunk, so a fact near a chunk boundary is not truncated out of
every chunk that could retrieve it.
"""

from dataclasses import dataclass

_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


@dataclass
class HeadingSection:
    heading_path: str
    body: str


def split_by_heading(raw: str) -> list[HeadingSection]:
    """Split a markdown document into sections at `#`-style headings."""
    sections: list[HeadingSection] = []
    heading = ""
    lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("#"):
            if lines:
                sections.append(HeadingSection(heading, "\n".join(lines)))
                lines = []
            heading = line.lstrip("#").strip()
        else:
            lines.append(line)
    if lines:
        sections.append(HeadingSection(heading, "\n".join(lines)))
    return sections


def _split_on_separator(text: str, separators: tuple[str, ...]) -> list[str]:
    """Recursively split `text` on the first separator that actually
    shortens the pieces, falling back to finer separators as needed."""
    if not separators:
        return [text]

    separator, *rest = separators
    if separator == "":
        return list(text)

    parts = text.split(separator)
    if len(parts) == 1:
        # This separator never occurs in the text — try a finer one.
        return _split_on_separator(text, rest)

    pieces: list[str] = []
    for part in parts:
        pieces.append(part)
    return pieces


class RecursiveCharacterTextSplitter:
    """Character-based recursive splitter with chunk overlap.

    This mirrors the well-established "recursive character splitting"
    strategy: try coarse separators first (paragraphs), fall back to finer
    ones (sentences, then words, then raw characters) only for pieces that
    are still too large, then greedily pack pieces into `chunk_size`-sized
    chunks with `chunk_overlap` characters repeated between consecutive
    chunks for context continuity.
    """

    def __init__(
        self,
        chunk_size: int = 1_000,
        chunk_overlap: int = 150,
        separators: tuple[str, ...] = _DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def _atomic_pieces(self, text: str) -> list[str]:
        """Break `text` into pieces no larger than `chunk_size`, splitting
        progressively finer only where a piece still doesn't fit."""
        pieces = _split_on_separator(text, self.separators)
        result: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                if piece:
                    result.append(piece)
                continue
            # Still too large after the coarsest applicable separator —
            # recurse with the remaining, finer separators.
            for finer_start, sep in enumerate(self.separators):
                if sep in piece or sep == "":
                    sub_pieces = _split_on_separator(piece, self.separators[finer_start:])
                    result.extend(p for p in sub_pieces if p)
                    break
        return result

    def split_text(self, text: str) -> list[str]:
        """Pack atomic pieces into chunks up to `chunk_size`, carrying the
        trailing `chunk_overlap` characters of one chunk into the next."""
        pieces = self._atomic_pieces(text.strip())
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
                overlap_tail = current[-self.chunk_overlap :]
                current = f"{overlap_tail} {piece}".strip()
            else:
                # A single piece already exceeds chunk_size (e.g. no spaces);
                # emit it as its own chunk rather than dropping content.
                chunks.append(piece[: self.chunk_size])
                current = piece[self.chunk_size - self.chunk_overlap :]
        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]


def chunk_document(
    raw: str,
    chunk_size: int = 1_000,
    chunk_overlap: int = 150,
) -> list[tuple[str, str]]:
    """Split a markdown document into (heading_path, chunk_content) pairs."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result: list[tuple[str, str]] = []
    for section in split_by_heading(raw):
        for piece in splitter.split_text(section.body):
            result.append((section.heading_path, piece))
    return result
