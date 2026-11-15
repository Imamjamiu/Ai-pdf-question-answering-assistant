"""Summarize manual answer and citation reviews without calling an LLM."""

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATION_PATH = PROJECT_DIR / "evaluation" / "questions.json"
REVIEW_PATH = PROJECT_DIR / "evaluation" / "answer_review.csv"
STATUSES = {"pass", "partial", "fail"}
FIELDS = [
    "question_id",
    "source_pdf",
    "question",
    "expected_pages",
    "answer_key",
    "answer_status",
    "citation_status",
    "review_notes",
]


def create_review_template():
    """Create a blank review sheet from the existing evaluation questions."""
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as review_file:
        writer = csv.DictWriter(review_file, fieldnames=FIELDS)
        writer.writeheader()

        for item in evaluation["questions"]:
            writer.writerow({
                "question_id": item["id"],
                "source_pdf": evaluation["source_pdf"],
                "question": item["question"],
                "expected_pages": ", ".join(
                    str(page) for page in item["expected_pages"]
                ),
                "answer_key": item.get("answer_key", ""),
                "answer_status": "",
                "citation_status": "",
                "review_notes": "",
            })

    print(f"Created review sheet: {REVIEW_PATH}")
    print("Fill in answer_status and citation_status with pass, partial, or fail.")


def load_review_rows():
    if not REVIEW_PATH.exists():
        create_review_template()
        return None

    with REVIEW_PATH.open("r", encoding="utf-8-sig", newline="") as review_file:
        return list(csv.DictReader(review_file))


def summarize_status(rows, field, label):
    reviewed = []
    invalid = []

    for row in rows:
        status = (row.get(field) or "").strip().lower()
        if not status:
            continue
        if status not in STATUSES:
            invalid.append((row.get("question_id", "?"), status))
        else:
            reviewed.append(status)

    print(f"\n{label}: {len(reviewed)}/{len(rows)} reviewed")
    if reviewed:
        counts = {status: reviewed.count(status) for status in ("pass", "partial", "fail")}
        pass_rate = counts["pass"] / len(reviewed)
        print(
            f"Pass: {counts['pass']} | Partial: {counts['partial']} | "
            f"Fail: {counts['fail']} | Pass rate: {pass_rate:.1%}"
        )
    else:
        print("No reviews recorded yet.")

    if invalid:
        print("Invalid status values (use pass, partial, or fail):")
        for question_id, status in invalid:
            print(f"  {question_id}: {status}")


def main():
    rows = load_review_rows()
    if rows is None:
        return

    print(f"Answer review for {len(rows)} questions")
    summarize_status(rows, "answer_status", "Answer accuracy and support")
    summarize_status(rows, "citation_status", "Citation accuracy")

    fully_reviewed = [
        row for row in rows
        if (row.get("answer_status") or "").strip().lower() in STATUSES
        and (row.get("citation_status") or "").strip().lower() in STATUSES
    ]
    both_pass = sum(
        (row.get("answer_status") or "").strip().lower() == "pass"
        and (row.get("citation_status") or "").strip().lower() == "pass"
        for row in fully_reviewed
    )
    if fully_reviewed:
        print(
            f"\nBoth answer and citations passed: {both_pass}/"
            f"{len(fully_reviewed)} fully reviewed questions"
        )
    else:
        print("\nNo questions have both ratings yet.")


if __name__ == "__main__":
    main()
