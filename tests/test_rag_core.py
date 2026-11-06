from rag_core import _chunks, extract_documents


def test_extract_pdf_rejects_unsupported_type():
    try:
        extract_documents("notes.csv", b"a,b")
    except Exception as error:
        assert "Unsupported file type" in str(error)
    else:
        raise AssertionError("Expected unsupported file type error")


def test_chunks_overlap_and_preserve_text():
    result = _chunks(" ".join(f"word{i}" for i in range(10)), size=5, overlap=2)
    assert len(result) == 3
    assert "word3" in result[1]
