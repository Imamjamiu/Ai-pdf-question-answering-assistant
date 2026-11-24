"""Generate grounded answers with OpenRouter first and Gemini as fallback."""
import json
import os
import httpx
import re
import time
from functools import lru_cache
from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
MAX_OUTPUT_TOKENS = 4096
# Provider quota state for the current Python process.
_OPENROUTER_DAILY_QUOTA_EXHAUSTED = False
_GEMINI_DAILY_QUOTA_EXHAUSTED = False

def _is_openrouter_daily_quota_error(error):
    if not isinstance(error, httpx.HTTPStatusError):
        return False

    if error.response.status_code != 429:
        return False

    response_text = error.response.text.lower()

    return "free-models-per-day" in response_text


def _is_gemini_daily_quota_error(error):
    error_code = getattr(error, "code", None)

    if error_code != 429 and "RESOURCE_EXHAUSTED" not in str(error).upper():
        return False

    error_message = str(error).lower()

    daily_quota_markers = (
        "perday",
        "requestsperday",
        "generate_content_free_tier_requests",
    )

    return any(marker in error_message for marker in daily_quota_markers)

@lru_cache(maxsize=1)
def load_gemini_client():
    """Create the Gemini client, requiring a key only when used."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)

def _format_passage(item):
    """Return passage text and an optional PDF page number."""
    if isinstance(item, dict):
        passage = item.get("text", "")
        page_number = item.get("page_number")
    elif isinstance(item, (tuple, list)) and item:
        passage = item[0]
        page_number = None
    elif isinstance(item, str):
        passage = item
        page_number = None
    else:
        return "", None
    if not isinstance(passage, str):
        return "", None
    return passage.strip(), page_number

def _number_passages(relevant_passages):
    """Assign stable IDs to the non-empty passages used in a prompt."""
    numbered_passages = []
    for item in relevant_passages:
        passage, page_number = _format_passage(item)
        if passage:
            passage_id = f"P{len(numbered_passages) + 1}"
            numbered_passages.append((passage_id, passage, page_number))
    return numbered_passages





def _format_visual_evidence(visual_evidence):
    """Format visual descriptions as separately identified evidence."""
    formatted_visuals = []

    for index, item in enumerate(visual_evidence or [], start=1):
        if not isinstance(item, dict):
            continue

        visual_id = item.get("visual_id") or f"V{index}"
        page_number = item.get("page_number")
        caption = item.get("caption", "")
        description = item.get("description", {})

        if not isinstance(description, dict):
            description = {}

        page_label = (
            f"PDF page {page_number}"
            if page_number is not None
            else "page number not provided"
        )

        parts = [
            f"[{visual_id} | {page_label}]",
            f"Caption: {caption}" if caption else "",
            f"Visual type: {description.get('visual_type', '')}",
            f"Summary: {description.get('summary', '')}",
        ]

        for field in (
            "key_elements",
            "relationships",
            "data_points",
            "uncertainties",
        ):
            value = description.get(field)

            if value:
                parts.append(f"{field.replace('_', ' ').title()}: {value}")

        formatted = "\n".join(str(part) for part in parts if part)
        formatted_visuals.append(formatted)

    return formatted_visuals

def _build_prompt(
    question,
    relevant_passages,
    conversation_history=None,
    visual_evidence=None,
):
    context_parts = []
    for passage_id, passage, page_number in _number_passages(relevant_passages):
        source_label = f"[{passage_id} | PDF page {page_number}]" if page_number else f"[{passage_id} | page number not provided]"
        context_parts.append(f"{source_label}\n{passage}")
    context = "\n\n".join(context_parts)
    visual_parts = _format_visual_evidence(visual_evidence)
    visual_context = "\n\n".join(visual_parts) or "No visual evidence provided."
    visual_analysis_unavailable = any(
        isinstance(item, dict)
        and isinstance(item.get("description"), dict)
        and item["description"].get("analysis_status") == "unavailable"
        for item in (visual_evidence or [])
    )

    if visual_analysis_unavailable:
        visual_reliability_instructions = """
    IMPORTANT: Visual analysis was unavailable for at least one supplied figure.
    Do not describe its colours, layout, labels, shapes, arrows, or other visual
    details as though you inspected them. Use supporting PDF text to explain
    the underlying concepts, but clearly disclose that the figure's visual
    details could not be verified. The original figure is displayed separately.
    Do not invent visual observations.
    """
    else:
        visual_reliability_instructions = """
    When visual analysis is available, use its description only for details
    supported by the visual evidence. Distinguish visible observations from
    technical explanations and preserve any stated uncertainties.
    """


    history_parts = []
    for message in (conversation_history or [])[-4:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "").strip().lower()
        content = message.get("content", "")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            history_parts.append(f"{role.title()}: {content[:600]}")
    history_context = "\n".join(history_parts) or "No earlier messages."
    return f"""
You are an academic PDF question-answering assistant. Answer using only the
supporting passages below. Read all passages and combine them when needed.

The conversation history is only for understanding references in the user's
current question, such as "that" or "the second point." It is not evidence.
Support every factual claim with the current supporting passages below. Do not
use previous answers or outside knowledge as sources.

Answer the user's full question in clear, natural language. Preserve the
source's important steps, conditions, assumptions, and exceptions. For a
process, distinguish required steps from conditional ones. Do not call a
sequence universal or complete unless the source supports that claim.
Answer every part of a multi-part question directly. If a conclusion follows
from the passage's explanation but is not stated directly, label it as an
inference and explain the passage's reasoning.
For a requested sequence, include all material steps from the stated starting
event through the outcome, in the order given by the source. Do not omit
state-saving, interrupt-handling, table-update, scheduling, or restart steps
when the passages include them. If a step is conditional, state its condition
and include any dependent step the source gives.

Do not invent facts or use outside knowledge. If the passages do not contain
enough information, say what is missing instead of guessing. Check that each
factual claim is supported before responding.

Write formulas in plain text so they display reliably. Avoid raw LaTeX; use
symbols such as × and ÷ where helpful, and explain variables in words. Keep
the answer complete and readable without adding unrelated detail.

CITATION RULES:

1. Cite every factual claim using the exact passage ID that supports it,
   such as [P1] or [P2].
2. Place citations immediately after the sentence or claim they support.
3. If a claim depends on multiple passages, cite all relevant passage IDs.
4. Do not cite a passage merely because it discusses the same topic.
   Its content must directly support the claim.
5. Never invent passage IDs, PDF page numbers, or citations.
6. Never write PDF page citations yourself. Only use passage IDs.
7. If the passages do not support a claim, do not attach a citation to it.
8. If the available passages cannot answer the question reliably, explain
   what information is missing rather than guessing.
9. Do not add a Sources section. The application handles citation conversion
   and the Sources section.

USER QUESTION:
{question}

CONVERSATION HISTORY (for context only, not evidence):
{history_context}

SUPPORTING TEXT PASSAGES:
{context}

SUPPORTING VISUAL EVIDENCE:
{visual_context}

VISUAL EVIDENCE RULES:

1. Visual evidence has IDs such as [V1] and text passages have IDs such as [P1].
2. Cite visual observations using their exact visual IDs.
3. Use visual evidence only for details that can reasonably be observed in the supplied description.
4. Distinguish directly observed visual details from interpretations.
5. Do not infer an algorithm's behaviour, sequence, or technical meaning from a diagram alone when the supporting text is needed.
6. Use the text passages to explain technical concepts and the visual evidence to describe what the figure actually shows.
7. If the visual description reports uncertainty, preserve that uncertainty rather than guessing.
8. Never invent visual IDs or claim that a figure shows something absent from its description.

{visual_reliability_instructions}

Now answer the question using the available supporting text passages and visual evidence.
""".strip()

def _extract_openrouter_text(payload):
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    answer = content.strip() if isinstance(content, str) else ""
    if not answer:
        raise ValueError("OpenRouter returned an empty response.")
    finish_reason = str(choices[0].get("finish_reason", "")).lower()
    if finish_reason == "length":
        answer += "\n\nNote: This answer may be incomplete because it reached the output length limit."
    return answer


def _normalize_citations(
    answer,
    relevant_passages,
    visual_evidence=None,
):
    """Convert valid text and visual IDs to PDF page citations."""

    evidence_pages = {}

    # Register text passage IDs.
    for passage_id, _, page_number in _number_passages(relevant_passages):
        try:
            if page_number is not None:
                evidence_pages[passage_id] = int(page_number)
        except (TypeError, ValueError):
            continue

    # Register visual evidence IDs.
    for index, item in enumerate(visual_evidence or [], start=1):
        if not isinstance(item, dict):
            continue

        visual_id = item.get("visual_id") or f"V{index}"
        page_number = item.get("page_number")

        try:
            if page_number is not None:
                evidence_pages[str(visual_id)] = int(page_number)
        except (TypeError, ValueError):
            continue

    citation_pattern = re.compile(
        r"\[((?:P|V|Passage)\s*\d+"
        r"(?:\s*(?:,|and|&)\s*(?:P|V|Passage)\s*\d+)*)\]",
        flags=re.IGNORECASE,
    )

    cited_pages = []

    def map_evidence_citations(match):
        evidence_ids = re.findall(
            r"(?:P|V|Passage)\s*\d+",
            match.group(1),
            flags=re.IGNORECASE,
        )

        pages = []

        for evidence_id in evidence_ids:
            normalized_id = re.sub(
                r"\s+",
                "",
                evidence_id,
            ).upper()

            page_number = evidence_pages.get(normalized_id)

            if page_number is not None and page_number not in pages:
                pages.append(page_number)

        for page in pages:
            if page not in cited_pages:
                cited_pages.append(page)

        return " ".join(
            f"[PDF page {page}]"
            for page in pages
        )

    # Remove model-generated PDF citations.
    answer = re.sub(
        r"\[PDF pages?\s+[^\]]*\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    answer = citation_pattern.sub(
        map_evidence_citations,
        answer,
    )

    # Remove any Sources heading generated by the model.
    answer = re.sub(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?Sources(?::)?(?:\*\*)?\s*:?.*$",
        "",
        answer,
    ).rstrip()

    answer = re.sub(
        r"(?<!\\)\*{3,}(?!\\)",
        "",
        answer,
    )

    if cited_pages:
        sources = "Sources: " + ", ".join(
            f"[PDF page {page}]"
            for page in cited_pages
        )
    else:
        sources = "Sources: No valid passage citations were included."

    return f"{answer}\n\n{sources}".strip()


def _generate_with_openrouter(prompt: str, max_retries: int = 3) -> str:
    """Generate an answer using OpenRouter with selective retries."""
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
    if not model:
        model = DEFAULT_OPENROUTER_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()
                return _extract_openrouter_text(response.json())

        except Exception as error:
            last_error = error

            if isinstance(error, httpx.HTTPStatusError):
                status_code = error.response.status_code

                # Retry only timeout and server-side HTTP errors.
                if status_code != 408 and status_code < 500:
                    raise

            elif isinstance(error, (ValueError, TypeError)):
                raise

            if attempt < max_retries:
                delay_seconds = 2 ** (attempt - 1)
                print(
                    f"OpenRouter request failed. "
                    f"Retrying in {delay_seconds} second(s). "
                    f"Error: {error}"
                )
                time.sleep(delay_seconds)
                continue

            raise last_error


def _generate_with_gemini(prompt):
    client = load_gemini_client()
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            answer = (response.text or "").strip()
            if not answer:
                raise ValueError("Gemini returned an empty response.")
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason).upper()
                if "MAX_TOKENS" in finish_reason:
                    answer += "\n\nNote: This answer may be incomplete because it reached the output length limit."
            return answer
        except Exception as error:
            error_code = getattr(error, "code", None)
            error_message = str(error).upper()
            is_unavailable = (error_code == 503 or "UNAVAILABLE" in error_message)
            if is_unavailable and attempt < 3:
                delay_seconds = 2 ** (attempt - 1)
                print(f"Gemini is unavailable; retrying in {delay_seconds} second(s).")
                time.sleep(delay_seconds)
                continue
            raise



def _generate_local_fallback(question, relevant_passages):
    """Build a basic extractive answer from retrieved PDF passages."""
    numbered_passages = _number_passages(relevant_passages)

    if not numbered_passages:
        return None

    question_terms = {
        term.lower()
        for term in re.findall(r"\b[a-zA-Z0-9]+\b", question)
        if len(term) > 2
    }

    stop_words = {
        "what", "which", "when", "where", "who", "why", "how",
        "does", "doesn't", "did", "is", "are", "was", "were",
        "the", "and", "for", "from", "with", "that", "this",
        "these", "those", "into", "about", "explain", "describe",
        "define", "give", "show", "tell", "can", "could",
        "would", "should", "you", "your"
    }

    question_terms -= stop_words

    scored_sentences = []

    for passage_id, passage, _page_number in numbered_passages:
        sentences = re.split(r"(?<=[.!?])\s+", passage)

        for sentence_index, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if len(sentence) < 30:
                continue

            sentence_terms = {
                term.lower()
                for term in re.findall(r"\b[a-zA-Z0-9]+\b", sentence)
                if len(term) > 2
            }

            overlap = question_terms & sentence_terms

            if overlap:
                score = sum(
                    2 if len(term) >= 7 else 1
                    for term in overlap
                )

                scored_sentences.append(
                    (score, sentence, passage_id, len(scored_sentences))
                )

    if not scored_sentences:
        # Preserve useful source text even when keyword overlap is low.
        passage_id, passage, _page_number = numbered_passages[0]
        selected = [(passage[:700].strip(), passage_id)]
    else:
        scored_sentences.sort(key=lambda item: item[0], reverse=True)

        selected = []
        seen_sentences = set()

        for _score, sentence, passage_id, position in scored_sentences:
            normalized = sentence.lower()

            if normalized in seen_sentences:
                continue

            seen_sentences.add(normalized)
            selected.append((sentence, passage_id, position))

            if len(selected) >= 5:
                break

        # Restore the original PDF sentence order.
        selected.sort(key=lambda item: item[2])

        # Remove the temporary position value.
        selected = [
            (sentence, passage_id)
            for sentence, passage_id, _position in selected
        ]

    answer_lines = [
        "### Answer (local extractive fallback)",
        "",
        "The cloud AI providers were unavailable. "
        "The following response is assembled directly from the retrieved "
        "PDF passages and has not been independently interpreted by an LLM.",
        "",
    ]

    for sentence, passage_id in selected:
        answer_lines.append(f"{sentence} [{passage_id}]")

    return "\n\n".join(answer_lines)


def _friendly_error(error, both_providers_failed=False):
    error_code = getattr(error, "code", None)
    error_message = str(error).upper()

    if both_providers_failed:
        return (
            "OpenRouter and Gemini are both unavailable right now, try again later."
        )

    if error_code == 429 or "RESOURCE_EXHAUSTED" in error_message:
        normalized_error = error_message.replace("_", "")
        if "PERDAY" in normalized_error or "REQUESTSPERDAY" in normalized_error:
            return "Gemini's daily request quota has been reached. The retrieved passages are still available below."
        if "PERMINUTE" in normalized_error or "RATELIMIT" in normalized_error:
            return "Gemini's short-term rate limit has been reached. Wait briefly, then try again."
        return "Gemini returned a quota or rate-limit error. Check the PyCharm Run console for details."
    if error_code == 503 or "UNAVAILABLE" in error_message:
        return "Gemini is temporarily unavailable. The retrieved passages are still available below; try again later."
    return "I couldn't generate an answer right now. Check the PyCharm Run console for provider details."

def generate_answer(
    question,
    relevant_passages,
    conversation_history=None,
    visual_evidence=None,
):
    global _OPENROUTER_DAILY_QUOTA_EXHAUSTED
    global _GEMINI_DAILY_QUOTA_EXHAUSTED

    if not isinstance(question, str) or not question.strip():

        return "Please provide a question."

    has_text_evidence = any(
        _format_passage(item)[0]
        for item in (relevant_passages or [])
    )

    has_visual_evidence = bool(visual_evidence)

    if not has_text_evidence and not has_visual_evidence:
        return "No relevant information was found."

    prompt = _build_prompt(
        question,
        relevant_passages or [],
        conversation_history=conversation_history,
        visual_evidence=visual_evidence,
    )

    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not openrouter_key and not gemini_key:
        return "No model API key is configured. Add OPENROUTER_API_KEY or GEMINI_API_KEY to your .env file."
    openrouter_error = (
        RuntimeError("OpenRouter daily free-model quota exhausted.")
        if _OPENROUTER_DAILY_QUOTA_EXHAUSTED
        else None
    )

    if openrouter_key and not _OPENROUTER_DAILY_QUOTA_EXHAUSTED:
        try:
            answer = _generate_with_openrouter(prompt)
            return _normalize_citations(
                answer,
                relevant_passages or [],
                visual_evidence=visual_evidence,
            )
        except Exception as error:
            openrouter_error = error
            if _is_openrouter_daily_quota_error(error):
                _OPENROUTER_DAILY_QUOTA_EXHAUSTED = True
                print("OpenRouter daily free-model quota exhausted.")
            print(f"OpenRouter API error: {error}")

    if gemini_key and not _GEMINI_DAILY_QUOTA_EXHAUSTED:
        try:
            answer = _generate_with_gemini(prompt)
            return _normalize_citations(
                answer,
                relevant_passages or [],
                visual_evidence=visual_evidence,
            )
        except Exception as error:
            if _is_gemini_daily_quota_error(error):
                _GEMINI_DAILY_QUOTA_EXHAUSTED = True
                print("Gemini daily request quota exhausted.")

            print(f"Gemini API error: {error}")

            return _friendly_error(
                error,
                both_providers_failed=openrouter_error is not None,
            )
    if _OPENROUTER_DAILY_QUOTA_EXHAUSTED and _GEMINI_DAILY_QUOTA_EXHAUSTED:
        return (
            "Both OpenRouter and Gemini daily quotas have been exhausted. "
            "The retrieved passages are still available below. "
            "Try again after the quotas reset."
        )

    if _OPENROUTER_DAILY_QUOTA_EXHAUSTED:
        return (
            "OpenRouter's daily free-model quota has been exhausted, "
            "and Gemini is not available. "
            "The retrieved passages are still available below."
        )

    if _GEMINI_DAILY_QUOTA_EXHAUSTED:
        return (
            "Gemini's daily request quota has been exhausted, "
            "and OpenRouter is not available. "
            "The retrieved passages are still available below."
        )

    print(f"OpenRouter API error: {openrouter_error}")
    return "No available provider could generate an answer. Check the PyCharm Run console for details."