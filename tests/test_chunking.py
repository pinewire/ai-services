from app.chunking import RecursiveCharacterTextSplitter, chunk_document, split_by_heading


def test_split_by_heading_groups_body_under_each_heading() -> None:
    raw = "# Alpha\nline one\nline two\n\n# Beta\nline three\n"
    sections = split_by_heading(raw)
    assert [s.heading_path for s in sections] == ["Alpha", "Beta"]
    assert "line one" in sections[0].body
    assert "line three" in sections[1].body


def test_splitter_respects_chunk_size() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    text = " ".join(f"word{i}" for i in range(100))
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_splitter_carries_overlap_between_chunks() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=15)
    text = " ".join(f"word{i}" for i in range(60))
    chunks = splitter.split_text(text)
    # Some trailing characters of one chunk should reappear at the start of the next.
    assert any(chunks[i][:5] in chunks[i - 1] for i in range(1, len(chunks)))


def test_splitter_rejects_overlap_not_smaller_than_chunk_size() -> None:
    try:
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_chunk_document_preserves_heading_and_splits_large_sections() -> None:
    raw = "# Runbook\n" + (" ".join(f"token{i}" for i in range(400)))
    chunks = chunk_document(raw, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(heading == "Runbook" for heading, _ in chunks)
