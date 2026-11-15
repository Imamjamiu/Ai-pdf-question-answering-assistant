"""Evaluate local retrieval for the bundled Virtual Memory PDF.

This script does not call Gemini. It measures expected-page coverage in the
same top-eight-plus-neighbor context that the Streamlit app sends to the model.
"""

import json
from pathlib import Path

from ai_pdf_assistant.pdf_processing import extract_pdf_pages
from ai_pdf_assistant.retrieval import (
    create_embeddings,
    create_keyword_index,
    find_relevant_passages,
    add_neighboring_chunks,
    split_pages_into_chunks,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_DIR / "evaluation" / "questions.json"
TOP_K = 8


def main():
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    pdf_path = PROJECT_DIR / evaluation["source_pdf"]

    if not pdf_path.exists():
        raise FileNotFoundError(f"Evaluation PDF not found: {pdf_path}")

    page_count, pages = extract_pdf_pages(pdf_path.read_bytes())
    chunks = split_pages_into_chunks(pages)
    embeddings = create_embeddings(chunks)
    keyword_index = create_keyword_index(chunks)

    print(f"PDF: {pdf_path.name} ({page_count} pages)")
    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(evaluation['questions'])}")
    print(
        f"Checking expected pages in the app context: top {TOP_K} retrieved "
        "chunks plus neighbors around the top two results.\n"
    )

    context_hit_count = 0
    reciprocal_rank_total = 0.0
    page_recall_total = 0.0

    for item in evaluation["questions"]:
        results = find_relevant_passages(
            question=item["question"],
            chunks=chunks,
            embeddings=embeddings,
            top_k=TOP_K,
            keyword_index=keyword_index,
        )
        context_passages = add_neighboring_chunks(results, chunks)

        expected_pages = set(item["expected_pages"])
        retrieved_pages = [result["page_number"] for result in context_passages]
        retrieved_page_set = set(retrieved_pages)
        matched_pages = expected_pages & retrieved_page_set

        if matched_pages:
            context_hit_count += 1

        first_match_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if result["page_number"] in expected_pages
            ),
            None,
        )
        if first_match_rank is not None:
            reciprocal_rank_total += 1 / first_match_rank

        page_recall_total += len(matched_pages) / len(expected_pages)

        status = "HIT" if matched_pages else "MISS"
        print(
            f"{status:4} {item['id']}: "
            f"expected={sorted(expected_pages)} "
            f"context_pages={retrieved_pages} "
            f"page_recall={len(matched_pages) / len(expected_pages):.2f}"
        )

    total = len(evaluation["questions"])
    print("\n=== Retrieval summary ===")
    print(
        f"Context hit rate: {context_hit_count}/{total} "
        f"({context_hit_count / total:.1%})"
    )
    print(
        "Mean expected-page recall in app context: "
        f"{page_recall_total / total:.1%}"
    )
    print(f"MRR@{TOP_K} (ranked chunks only): {reciprocal_rank_total / total:.3f}")
    print("Gemini was not called; this evaluates retrieval only.")


if __name__ == "__main__":
    main()
