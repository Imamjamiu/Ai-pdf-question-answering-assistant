import numpy as np
import pytest

from ai_pdf_assistant import retrieval


class FixedEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=float)


def test_empty_text_produces_no_chunks():
    assert retrieval.split_text_into_chunks("") == []


def test_split_pages_preserves_page_numbers_and_page_boundaries():
    pages = [
        {"page_number": 2, "text": "First sentence. Second sentence."},
        {"page_number": 3, "text": "Another page."},
    ]

    chunks = retrieval.split_pages_into_chunks(
        pages,
        chunk_size=40,
        overlap=0,
    )

    assert [chunk["page_number"] for chunk in chunks] == [2, 3]
    assert chunks[0]["text"] == "First sentence. Second sentence."
    assert chunks[1]["text"] == "Another page."


def test_get_chunk_text_accepts_text_and_page_chunks():
    assert retrieval.get_chunk_text("plain text") == "plain text"
    assert retrieval.get_chunk_text({"text": "page text", "page_number": 4}) == "page text"


def test_get_chunk_text_rejects_a_chunk_without_text():
    with pytest.raises(TypeError):
        retrieval.get_chunk_text({"page_number": 4})


def test_neighbor_expansion_deduplicates_and_sorts_by_chunk_index():
    chunks = [
        {"text": f"chunk {index}", "page_number": 1}
        for index in range(5)
    ]
    relevant = [
        {
            "text": "chunk 2",
            "page_number": 1,
            "hybrid_score": 0.9,
            "semantic_rank": 1,
            "keyword_rank": 1,
            "chunk_index": 2,
        }
    ]

    expanded = retrieval.add_neighboring_chunks(relevant, chunks)

    assert [item["chunk_index"] for item in expanded] == [1, 2, 3]
    assert [item["is_neighbor"] for item in expanded] == [True, False, True]
    assert expanded[1]["hybrid_score"] == 0.9


def test_hybrid_search_can_promote_a_keyword_match(monkeypatch):
    chunks = [
        {"text": "alpha information", "page_number": 1},
        {"text": "beta information", "page_number": 2},
    ]
    embeddings = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    keyword_index = retrieval.create_keyword_index(chunks)
    monkeypatch.setattr(
        retrieval,
        "load_embedding_model",
        lambda: FixedEmbeddingModel(),
    )

    results = retrieval.find_relevant_passages(
        question="alpha?",
        chunks=chunks,
        embeddings=embeddings,
        top_k=1,
        keyword_index=keyword_index,
    )

    assert len(results) == 1
    assert results[0]["chunk_index"] == 0
    assert results[0]["keyword_rank"] == 1

