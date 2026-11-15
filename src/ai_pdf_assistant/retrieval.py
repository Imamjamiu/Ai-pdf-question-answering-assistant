
import re
import numpy as np

from functools import lru_cache


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_embedding_model():

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


# --------------------------------------------------
# 2. Get the text from a chunk
# --------------------------------------------------

def get_chunk_text(chunk):

    # Support ordinary text chunks.
    if isinstance(chunk, str):
        return chunk

    # Support page-aware dictionary chunks.
    if isinstance(chunk, dict):

        text = chunk.get("text")

        if isinstance(text, str):
            return text

    raise TypeError(
        "Each chunk must be a string or a dictionary containing text."
    )


# --------------------------------------------------
# 3. Divide document text into smaller chunks
# --------------------------------------------------

def split_text_into_chunks(text, chunk_size=1000, overlap=150):

    # Repair common PDF line-break hyphenation.
    text = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Split text into sentences.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    units = []

    # Handle unusually long sentences.
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

    # Combine sentences into chunks.
    chunks = []
    current_chunk = []

    def calculate_length(parts):

        return (
            sum(len(part) for part in parts)
            + max(0, len(parts) - 1)
        )

    for sentence in units:

        if (
            current_chunk
            and calculate_length(current_chunk)
            + len(sentence) + 1 > chunk_size
        ):

            # Save the completed chunk.
            chunks.append(" ".join(current_chunk))

            # Preserve whole sentences for overlap.
            overlap_parts = []
            overlap_length = 0

            for previous_sentence in reversed(current_chunk):

                added_length = len(previous_sentence) + (
                    1 if overlap_parts else 0
                )

                if overlap_length + added_length > overlap:
                    break

                overlap_parts.insert(0, previous_sentence)
                overlap_length += added_length

            current_chunk = overlap_parts

            # Remove old overlap if necessary.
            while (
                current_chunk
                and calculate_length(current_chunk)
                + len(sentence) + 1 > chunk_size
            ):

                current_chunk.pop(0)

        current_chunk.append(sentence)

    # Save remaining text.
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# --------------------------------------------------
# 4. Split PDF pages into chunks with page metadata
# --------------------------------------------------

def split_pages_into_chunks(
    pages,
    chunk_size=1000,
    overlap=150
):

    page_chunks = []

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"]

        chunks = split_text_into_chunks(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:

            page_chunks.append({
                "text": chunk,
                "page_number": page_number
            })

    return page_chunks


# --------------------------------------------------
# 5. Convert chunks into semantic embeddings
# --------------------------------------------------

def create_embeddings(chunks):

    if not chunks:
        return []

    # Extract text, even when chunks contain metadata.
    texts = [
        get_chunk_text(chunk)
        for chunk in chunks
    ]

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings


# --------------------------------------------------
# 6. Create keyword index using TF-IDF
# --------------------------------------------------

def create_keyword_index(chunks):

    if not chunks:
        return None

    texts = [
        get_chunk_text(chunk)
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:

        chunk_vectors = vectorizer.fit_transform(texts)

    except ValueError:

        return None

    return vectorizer, chunk_vectors


# --------------------------------------------------
# 7. Hybrid retrieval using Reciprocal Rank Fusion
# --------------------------------------------------

def find_relevant_passages(
    question,
    chunks,
    embeddings,
    top_k=3,
    keyword_index=None,
    rrf_constant=60,
    min_semantic_score=0.30,
    min_keyword_score=0.05
):

    if not isinstance(question, str) or not question.strip():
        return []

    if not chunks:
        return []

    if top_k <= 0:
        return []

    if embeddings is None or len(embeddings) != len(chunks):

        raise ValueError(
            "The number of embeddings must match the number of chunks."
        )

    # ----------------------------------------------
    # A. Semantic retrieval
    # ----------------------------------------------

    texts = [
        get_chunk_text(chunk)
        for chunk in chunks
    ]

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    semantic_scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    semantic_order = np.argsort(
        -semantic_scores,
        kind="stable"
    )

    semantic_ranks = np.empty(
        len(chunks),
        dtype=int
    )

    semantic_ranks[semantic_order] = np.arange(
        1,
        len(chunks) + 1
    )

    # ----------------------------------------------
    # B. Keyword retrieval
    # ----------------------------------------------

    if keyword_index is None:
        keyword_index = create_keyword_index(chunks)

    keyword_ranks = {}
    keyword_scores = np.zeros(len(chunks), dtype=float)

    if keyword_index is not None:

        vectorizer, chunk_vectors = keyword_index

        question_vector = vectorizer.transform(
            [question]
        )

        keyword_scores = cosine_similarity(
            question_vector,
            chunk_vectors
        ).flatten()

        # Ignore chunks with no keyword match.
        matching_indices = np.flatnonzero(
            keyword_scores > 0
        )

        keyword_order = matching_indices[
            np.argsort(
                -keyword_scores[matching_indices],
                kind="stable"
            )
        ]

        for rank, index in enumerate(keyword_order, start=1):

            keyword_ranks[int(index)] = rank

    # ----------------------------------------------
    # C. Combine rankings using RRF
    # ----------------------------------------------

    hybrid_results = []

    for index, chunk in enumerate(chunks):

        semantic_rank = int(semantic_ranks[index])
        semantic_score = float(semantic_scores[index])
        keyword_score = float(keyword_scores[index])

        # RRF always assigns a score based on rank, even to unrelated text.
        # Keep a passage only when at least one underlying retrieval signal
        # clears its relevance threshold. Calibrate these defaults against
        # representative questions for the document collection.
        if (
            semantic_score < min_semantic_score
            and keyword_score < min_keyword_score
        ):
            continue

        hybrid_score = 1 / (
            rrf_constant + semantic_rank
        )

        keyword_rank = keyword_ranks.get(index)

        if keyword_rank is not None:

            hybrid_score += 1 / (
                rrf_constant + keyword_rank
            )

        hybrid_results.append({
            "text": texts[index],
            "page_number": (
                chunk.get("page_number")
                if isinstance(chunk, dict)
                else None
            ),
            "hybrid_score": float(hybrid_score),
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "semantic_rank": semantic_rank,
            "keyword_rank": keyword_rank,
            "chunk_index": index
        })

    # ----------------------------------------------
    # D. Sort by hybrid score
    # ----------------------------------------------

    hybrid_results.sort(
        key=lambda item: (
            -item["hybrid_score"],
            item["chunk_index"]
        )
    )

    # ----------------------------------------------
    # E. Return the top relevant passages
    # ----------------------------------------------

    return hybrid_results[:top_k]

def add_neighboring_chunks(relevant_passages, chunks, neighbor_count=2):

    expanded_passages = []
    seen_indices = set()

    # Keep the original retrieved passages first
    for result in relevant_passages:
        item = result.copy()
        item["is_neighbor"] = False

        expanded_passages.append(item)
        seen_indices.add(item["chunk_index"])

    # Add neighbouring chunks around the highest-ranked results.
    for result in relevant_passages[:max(0, neighbor_count)]:

        current_index = result["chunk_index"]

        neighboring_indices = [
            current_index - 1,
            current_index + 1
        ]

        for neighbor_index in neighboring_indices:

            if 0 <= neighbor_index < len(chunks):

                if neighbor_index not in seen_indices:

                    chunk = chunks[neighbor_index]

                    expanded_passages.append({
                        "text": get_chunk_text(chunk),
                        "page_number": (
                            chunk.get("page_number")
                            if isinstance(chunk, dict)
                            else None
                        ),
                        "chunk_index": neighbor_index,
                        "is_neighbor": True
                    })

                    seen_indices.add(neighbor_index)
                    # Arrange all passages in their original PDF order
                    expanded_passages.sort(
                        key=lambda item: item["chunk_index"]
                    )

    return expanded_passages
