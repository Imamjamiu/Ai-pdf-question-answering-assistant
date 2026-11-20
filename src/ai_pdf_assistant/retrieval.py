import os
import re
import numpy as np
from functools import lru_cache
from typing import Any
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def load_embedding_model(model_name: str | None = None):
    """Load and cache the sentence-transformer embedding model."""
    from sentence_transformers import SentenceTransformer
    name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return SentenceTransformer(name)

def get_chunk_text(chunk: str | dict[str, Any]) -> str:
    """Extract the text content from a chunk."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        text = chunk.get("text")
        if isinstance(text, str):
            return text
    raise TypeError("Each chunk must be a string or a dictionary containing text.")

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks while trying to respect sentence boundaries."""
    text = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    units = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= chunk_size:
            units.append(sentence)
        else:
            words = sentence.split()
            current_piece = []
            for word in words:
                candidate = " ".join(current_piece + [word])
                if len(candidate) > chunk_size and current_piece:
                    units.append(" ".join(current_piece))
                    current_piece = [word]
                else:
                    current_piece.append(word)
            if current_piece:
                units.append(" ".join(current_piece))
    chunks = []
    current_chunk = []
    def calculate_length(parts):
        return sum(len(part) for part in parts) + max(0, len(parts) - 1)
    for sentence in units:
        if current_chunk and calculate_length(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_parts = []
            overlap_length = 0
            for previous_sentence in reversed(current_chunk):
                added_length = len(previous_sentence) + (1 if overlap_parts else 0)
                if overlap_length + added_length > overlap:
                    break
                overlap_parts.insert(0, previous_sentence)
                overlap_length += added_length
            current_chunk = overlap_parts
            while current_chunk and calculate_length(current_chunk) + len(sentence) + 1 > chunk_size:
                current_chunk.pop(0)
        current_chunk.append(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def split_pages_into_chunks(pages: list[dict[str, Any]], chunk_size: int = 1000, overlap: int = 150) -> list[dict[str, Any]]:
    """Split a list of pages into page-aware text chunks."""
    page_chunks = []
    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]
        chunks = split_text_into_chunks(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            page_chunks.append({"text": chunk, "page_number": page_number})
    return page_chunks

def create_embeddings(chunks: list[str | dict[str, Any]]) -> NDArray[np.floating] | list:
    """Convert chunks into semantic embeddings."""
    if not chunks:
        return []
    texts = [get_chunk_text(chunk) for chunk in chunks]
    model = load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

def create_keyword_index(chunks):
    """Create keyword index using TF-IDF."""
    if not chunks:
        return None
    texts = [get_chunk_text(chunk) for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        chunk_vectors = vectorizer.fit_transform(texts)
    except ValueError:
        return None
    return vectorizer, chunk_vectors

def find_relevant_passages(
    question: str,
    chunks: list[str | dict[str, Any]],
    embeddings: NDArray[np.floating],
    top_k: int = 3,
    keyword_index: tuple | None = None,
    rrf_constant: int = 60,
    min_semantic_score: float = 0.30,
    min_keyword_score: float = 0.05,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant passages using hybrid semantic + keyword search (RRF)."""
    if not isinstance(question, str) or not question.strip():
        return []
    if not chunks:
        return []
    if top_k <= 0:
        return []
    if embeddings is None or len(embeddings) != len(chunks):
        raise ValueError("The number of embeddings must match the number of chunks.")
    texts = [get_chunk_text(chunk) for chunk in chunks]
    model = load_embedding_model()
    question_embedding = model.encode([question], convert_to_numpy=True, show_progress_bar=False)
    semantic_scores = cosine_similarity(question_embedding, embeddings)[0]
    semantic_order = np.argsort(-semantic_scores, kind="stable")
    semantic_ranks = np.empty(len(chunks), dtype=int)
    semantic_ranks[semantic_order] = np.arange(1, len(chunks) + 1)
    if keyword_index is None:
        keyword_index = create_keyword_index(chunks)
    keyword_ranks = {}
    keyword_scores = np.zeros(len(chunks), dtype=float)
    if keyword_index is not None:
        vectorizer, chunk_vectors = keyword_index
        question_vector = vectorizer.transform([question])
        keyword_scores = cosine_similarity(question_vector, chunk_vectors).flatten()
        matching_indices = np.flatnonzero(keyword_scores > 0)
        keyword_order = matching_indices[np.argsort(-keyword_scores[matching_indices], kind="stable")]
        for rank, index in enumerate(keyword_order, start=1):
            keyword_ranks[int(index)] = rank
    hybrid_results = []
    for index, chunk in enumerate(chunks):
        semantic_rank = int(semantic_ranks[index])
        semantic_score = float(semantic_scores[index])
        keyword_score = float(keyword_scores[index])
        if semantic_score < min_semantic_score and keyword_score < min_keyword_score:
            continue
        hybrid_score = 1 / (rrf_constant + semantic_rank)
        keyword_rank = keyword_ranks.get(index)
        if keyword_rank is not None:
            hybrid_score += 1 / (rrf_constant + keyword_rank)
        hybrid_results.append({
            "text": texts[index],
            "page_number": chunk.get("page_number") if isinstance(chunk, dict) else None,
            "hybrid_score": float(hybrid_score),
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "semantic_rank": semantic_rank,
            "keyword_rank": keyword_rank,
            "chunk_index": index
        })
    hybrid_results.sort(key=lambda item: (-item["hybrid_score"], item["chunk_index"]))
    return hybrid_results[:top_k]

def add_neighboring_chunks(
    relevant_passages: list[dict[str, Any]],
    chunks: list[str | dict[str, Any]],
    neighbor_count: int = 2,
) -> list[dict[str, Any]]:
    """Expand retrieved passages with neighboring chunks for better context."""
    expanded_passages = []
    seen_indices = set()
    for result in relevant_passages:
        item = result.copy()
        item["is_neighbor"] = False
        expanded_passages.append(item)
        seen_indices.add(item["chunk_index"])
    for result in relevant_passages[: max(0, neighbor_count)]:
        current_index = result["chunk_index"]
        for neighbor_index in (current_index - 1, current_index + 1):
            if 0 <= neighbor_index < len(chunks) and neighbor_index not in seen_indices:
                chunk = chunks[neighbor_index]
                expanded_passages.append({
                    "text": get_chunk_text(chunk),
                    "page_number": chunk.get("page_number") if isinstance(chunk, dict) else None,
                    "chunk_index": neighbor_index,
                    "is_neighbor": True,
                })
                seen_indices.add(neighbor_index)
    expanded_passages.sort(key=lambda item: item["chunk_index"])
    return expanded_passages