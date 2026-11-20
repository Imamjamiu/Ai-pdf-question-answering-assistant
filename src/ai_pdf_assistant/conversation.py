"""Conversation and answer-display helpers for the Streamlit UI."""
import re
from typing import Any

def is_acknowledgement(message: str) -> bool:
    """Check whether a message is a simple conversational acknowledgement."""
    acknowledgements = {
        "thanks", "thank you", "thanks a lot", "thank you very much", "okay", "ok", "okay thanks", "ok thanks",
        "i understand", "understood", "got it", "good", "great", "nice", "perfect", "that's helpful",
        "that is helpful", "very helpful", "helpful", "alright", "all right", "cool", "noted", "makes sense",
        "i see", "excellent", "well done",
    }
    normalized = re.sub(r"[.!?,]+$", "", message.lower().strip())
    return normalized in acknowledgements

def is_summary_follow_up(question: str) -> bool:
    """Recognize requests to summarize the previous answer."""
    return bool(re.match(
        r"^\s*(?:please\s+)?(?:summari[sz]e|shorten)\s+"
        r"(?:(?:your|that|this|the previous|the last)\s+)?"
        r"(?:answer|response|explanation|it|that|this)\s*[?.!]*$",
        question, flags=re.IGNORECASE,
    ))

def is_contextual_follow_up(question: str, active_topic: str | None) -> bool:
    """Identify incomplete references that need the current chat topic."""
    if not active_topic or not isinstance(question, str):
        return False
    question = question.strip()
    if not question:
        return False
    explicit_follow_up_patterns = (
        r"^(?:why|how|what|who|when|where)\s*[?.!]*$",
        r"^(?:tell me more|say more|go into more detail|give me an example|another example)\b",
        r"^(?:explain|elaborate|expand|clarify)\s+(?:that|this|it|more|further)\b",
        r"^(?:can|could)\s+you\s+(?:explain|elaborate|expand|clarify)\s+(?:that|this|it|more|further)\b",
        r"^(?:what|how)\s+about\s+(?:that|this|it|its|their)\b",
        r"^(?:make|keep)\s+(?:it|that|this)\s+(?:shorter|simpler|clearer)\b",
        r"^(?:why|how|what)\s+(?:is|does|do|did|was|were|are|can|could|should|would|must)\s+(?:it|that|this|they|those|these)\b",
        r"^(?:and|also|then)\s+(?:why|how|what|can you|could you|explain|give me)\b",
        r"^(?:how is that|how does that|why is that|why does that)\b",
        r"^(?:how is it|how does it|why is it|why does it)\b",
    )
    if not any(re.search(p, question, flags=re.IGNORECASE) for p in explicit_follow_up_patterns):
        return False
    generic_words = {
        "a", "about", "also", "an", "and", "are", "as", "can", "could", "did", "does", "do", "explain", "further",
        "give", "how", "i", "is", "it", "me", "more", "of", "please", "that", "the", "their", "them", "then",
        "this", "to", "what", "when", "where", "which", "who", "why", "with", "would", "you",
    }
    content_words = [
        word.lower() for word in re.findall(r"[A-Za-z0-9]+", question) if word.lower() not in generic_words
    ]
    return len(content_words) <= 3

def build_retrieval_query(question: str, active_topic: str | None = None) -> str:
    """Add the current topic only when the question is an incomplete follow-up."""
    if not active_topic:
        return question
    return f"Recent topic: {active_topic[:400]}\nCurrent follow-up question: {question}"

def get_cited_pages(answer: str, context_passages: list[dict[str, Any]]) -> list[str]:
    """Return cited PDF pages that have matching retrieved passages."""
    available_pages = {
        str(result.get("page_number")) for result in context_passages if result.get("page_number") is not None
    }
    cited_pages = []
    for page_number in re.findall(r"\[PDF page\s+(\d+)\]", answer, flags=re.IGNORECASE):
        if page_number in available_pages and page_number not in cited_pages:
            cited_pages.append(page_number)
    return cited_pages

def remove_sources_line(answer: str) -> str:
    """Hide the generated Sources line because page controls replace it."""
    return re.sub(r"(?im)^\s*Sources\s*:.*$", "", answer).strip()

def clean_answer_for_display(answer: str) -> str:
    """Hide inline citation tokens because compact page controls replace them."""
    answer = remove_sources_line(answer)
    answer = re.sub(r"\[PDF page\s+\d+\]", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?<!\*)\*{3,}(?!\*)", "", answer)
    answer = re.sub(r"(?<=\w) {2,}(?=\w)", " ", answer)
    answer = re.sub(r"[ \t]+([,.;:!?])", r"\1", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()