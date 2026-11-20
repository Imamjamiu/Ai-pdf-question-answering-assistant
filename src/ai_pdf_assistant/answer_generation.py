"""Generate grounded answers with OpenRouter first and Gemini as fallback."""
import json
import os
import httpx
import re
import time
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
MAX_OUTPUT_TOKENS = 4096

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

def _build_prompt(question, relevant_passages, conversation_history=None):
    context_parts = []
    for passage_id, passage, page_number in _number_passages(relevant_passages):
        source_label = f"[{passage_id} | PDF page {page_number}]" if page_number else f"[{passage_id} | page number not provided]"
        context_parts.append(f"{source_label}\n{passage}")
    context = "\n\n".join(context_parts)
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

SUPPORTING PASSAGES:
{context}

Now answer the question from these passages.
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

def _normalize_citations(answer, relevant_passages):
    """Convert valid passage IDs to PDF page citations and add a Sources line."""
    passage_pages = {}
    for passage_id, _, page_number in _number_passages(relevant_passages):
        try:
            if page_number is not None:
                passage_pages[passage_id] = int(page_number)
        except (TypeError, ValueError):
            continue
    passage_citation_pattern = re.compile(
        r"\[(?:(?:P|Passage)\s*\d+)"
        r"(?:\s*(?:,|and|&)\s*(?:P|Passage)\s*\d+)*\]",
        flags=re.IGNORECASE,
    )
    cited_pages = []
    def map_passage_citations(match):
        passage_ids = re.findall(r"(?:P|Passage)\s*(\d+)", match.group(0), flags=re.IGNORECASE)
        pages = []
        for passage_number in passage_ids:
            page_number = passage_pages.get(f"P{passage_number}")
            if page_number is not None and page_number not in pages:
                pages.append(page_number)
        for page in pages:
            if page not in cited_pages:
                cited_pages.append(page)
        return " ".join(f"[PDF page {page}]" for page in pages)
    answer = re.sub(r"\[PDF pages?\s+[^\]]*\]", "", answer, flags=re.IGNORECASE)
    answer = passage_citation_pattern.sub(map_passage_citations, answer)
    answer = re.sub(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?Sources(?::)?(?:\*\*)?\s*:?.*$", "", answer).rstrip()
    answer = re.sub(r"(?<!\\)\*{3,}(?!\\)", "", answer)
    if cited_pages:
        sources = "Sources: " + ", ".join(f"[PDF page {page}]" for page in cited_pages)
    else:
        sources = "Sources: No valid passage citations were included."
    return f"{answer}\n\n{sources}".strip()

def _generate_with_openrouter(prompt: str, max_retries: int = 3) -> str:
    """Generate an answer using OpenRouter with simple retries."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
    if not model:
        model = DEFAULT_OPENROUTER_MODEL
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.2}
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                return _extract_openrouter_text(response.json())
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
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

def _friendly_error(error, both_providers_failed=False):
    error_code = getattr(error, "code", None)
    error_message = str(error).upper()
    if error_code == 429 or "RESOURCE_EXHAUSTED" in error_message:
        normalized_error = error_message.replace("_", "")
        if "PERDAY" in normalized_error or "REQUESTSPERDAY" in normalized_error:
            return "Gemini's daily request quota has been reached. The retrieved passages are still available below."
        if "PERMINUTE" in normalized_error or "RATELIMIT" in normalized_error:
            return "Gemini's short-term rate limit has been reached. Wait briefly, then try again."
        return "Gemini returned a quota or rate-limit error. Check the PyCharm Run console for details."
    if both_providers_failed:
        return "OpenRouter and Gemini are both unavailable right now. The retrieved passages are still available below; try again later."
    if error_code == 503 or "UNAVAILABLE" in error_message:
        return "Gemini is temporarily unavailable. The retrieved passages are still available below; try again later."
    return "I couldn't generate an answer right now. Check the PyCharm Run console for provider details."

def generate_answer(question, relevant_passages, conversation_history=None):
    if not isinstance(question, str) or not question.strip():
        return "Please provide a question."
    if not relevant_passages:
        return "No relevant information was found."
    if not any(_format_passage(item)[0] for item in relevant_passages):
        return "No usable supporting passage text was provided."
    prompt = _build_prompt(question, relevant_passages, conversation_history=conversation_history)
    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not openrouter_key and not gemini_key:
        return "No model API key is configured. Add OPENROUTER_API_KEY or GEMINI_API_KEY to your .env file."
    openrouter_error = None
    if openrouter_key:
        try:
            answer = _generate_with_openrouter(prompt)
            return _normalize_citations(answer, relevant_passages)
        except Exception as error:
            openrouter_error = error
            print(f"OpenRouter API error: {error}")
    if gemini_key:
        try:
            answer = _generate_with_gemini(prompt)
            return _normalize_citations(answer, relevant_passages)
        except Exception as error:
            print(f"Gemini API error: {error}")
            return _friendly_error(error, both_providers_failed=openrouter_error is not None)
    print(f"OpenRouter API error: {openrouter_error}")
    return "OpenRouter could not generate an answer. Check the PyCharm Run console for details."