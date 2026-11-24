import pytest
from src.ai_pdf_assistant.answer_generation import (
    _build_prompt,
    _normalize_citations,
    _generate_local_fallback,
    generate_answer,
)

# Fixture to reset provider quota state before and after each test
@pytest.fixture(autouse=True)
def reset_provider_quota_state():
    from src.ai_pdf_assistant import answer_generation

    answer_generation._OPENROUTER_DAILY_QUOTA_EXHAUSTED = False
    answer_generation._GEMINI_DAILY_QUOTA_EXHAUSTED = False

    yield

    answer_generation._OPENROUTER_DAILY_QUOTA_EXHAUSTED = False
    answer_generation._GEMINI_DAILY_QUOTA_EXHAUSTED = False


# Tests for citation normalization
def test_existing_text_passage_citations_are_preserved():
    """Ensure valid passage citations are preserved and formatted correctly."""
    passages = [
        {"text": "Virtual memory maps virtual addresses to physical memory.", "page_number": 3},
        {"text": "The backing store holds pages that are not currently in RAM.", "page_number": 4},
    ]
    answer = "Virtual memory uses address mapping [P1]. The backing store holds inactive pages [P2]."
    result = _normalize_citations(answer, passages)

    assert "[PDF page 3]" in result
    assert "[PDF page 4]" in result
    assert "[P1]" not in result
    assert "[P2]" not in result


def test_invalid_passage_citations_are_removed():
    """Ensure invalid passage citations are removed."""
    passages = [{"text": "Virtual memory maps addresses.", "page_number": 3}]
    answer = "Virtual memory maps addresses [P5]."
    result = _normalize_citations(answer, passages)

    assert "[P5]" not in result
    assert "[PDF page 3]" not in result
    assert "Sources: No valid passage citations were included." in result


# Tests for prompt generation
def test_prompt_includes_passage_ids_and_page_numbers():
    """Ensure prompts include passage IDs and page numbers."""
    passages = [{"text": "Page replacement selects a page to remove from memory.", "page_number": 21}]
    prompt = _build_prompt("What is page replacement?", passages)

    assert "[P1 | PDF page 21]" in prompt
    assert "What is page replacement?" in prompt
    assert "Do not invent facts" in prompt


def test_visual_evidence_is_included_in_prompt():
    """Ensure visual evidence is included in the prompt."""
    passages = [{"text": "The clock algorithm uses a reference bit.", "page_number": 23}]
    visual_evidence = [
        {
            "visual_id": "V1",
            "caption": "Figure 10.17: Second-chance page-replacement algorithm",
            "page_number": 23,
            "description": {
                "visual_type": "diagram",
                "summary": "A circular arrangement of page frames and reference bits.",
                "key_elements": ["Page frames", "Reference bits", "Clock pointer"],
                "uncertainties": [],
            },
        }
    ]
    prompt = _build_prompt("Explain Figure 10.17.", passages, visual_evidence=visual_evidence)

    assert "[V1 | PDF page 23]" in prompt
    assert "Second-chance page-replacement algorithm" in prompt
    assert "reference bits" in prompt


def test_unavailable_visual_analysis_adds_reliability_warning():
    """Ensure unavailable visual analysis adds a reliability warning."""
    passages = [{"text": "Virtual memory maps pages to physical memory.", "page_number": 3}]
    visual_evidence = [
        {
            "visual_id": "V1",
            "caption": "Figure 10.1: Virtual memory",
            "page_number": 3,
            "description": {"analysis_status": "unavailable", "summary": "AI visual analysis is temporarily unavailable."},
        }
    ]
    prompt = _build_prompt("Explain Figure 10.1.", passages, visual_evidence=visual_evidence)

    assert "Visual analysis was unavailable" in prompt
    assert "Do not describe its colours" in prompt
    assert "Do not invent visual observations" in prompt


# Tests for local fallback generation
def test_local_fallback_extracts_relevant_sentences():
    """Ensure local fallback extracts relevant sentences."""
    passages = [
        {
            "text": (
                "Virtual memory allows processes to use an address space larger than physical memory. "
                "The memory management unit translates virtual addresses into physical addresses."
            ),
            "page_number": 3,
        }
    ]
    result = _generate_local_fallback("What is virtual memory?", passages)

    assert result is not None
    assert "[P1]" in result
    assert "Virtual memory allows processes" in result
    assert "local extractive fallback" in result


def test_local_fallback_preserves_original_pdf_sentence_order():
    """Ensure local fallback preserves the original PDF sentence order."""
    passages = [
        {
            "text": (
                "Virtual memory is a technique used by operating systems to manage memory efficiently. "
                "The memory management unit translates virtual addresses into physical addresses."
            ),
            "page_number": 3,
        }
    ]
    result = _generate_local_fallback(
        "Explain virtual memory, physical addresses and memory translation.", passages
    )

    first_sentence = "Virtual memory is a technique"
    second_sentence = "The memory management unit"

    assert first_sentence in result
    assert second_sentence in result
    assert result.index(first_sentence) < result.index(second_sentence)


def test_local_fallback_prioritizes_sentences_with_technical_terms():
    """Ensure local fallback prioritizes sentences with technical terms."""
    passages = [
        {
            "text": (
                "The system uses memory for several important operations. "
                "Virtual memory management uses page tables to translate virtual addresses into physical addresses efficiently."
            ),
            "page_number": 3,
        }
    ]
    result = _generate_local_fallback(
        "Explain virtual memory management and address translation.", passages
    )

    technical_sentence = (
        "Virtual memory management uses page tables to translate virtual addresses into physical addresses efficiently."
    )

    assert technical_sentence in result


# Tests for error handling in `generate_answer`
def test_generate_answer_reports_error_when_providers_fail(monkeypatch):
    """Ensure `generate_answer` reports an error when both providers fail."""
    from src.ai_pdf_assistant import answer_generation

    def fail_provider(_prompt):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(answer_generation, "_generate_with_openrouter", fail_provider)
    monkeypatch.setattr(answer_generation, "_generate_with_gemini", fail_provider)

    passages = [
        {"text": "Virtual memory allows processes to use an address space larger than physical memory.", "page_number": 3}
    ]
    result = generate_answer("What is virtual memory?", passages)

    assert "local extractive fallback" not in result
    assert "OpenRouter and Gemini are both unavailable" in result


def test_generate_answer_reports_error_when_gemini_key_is_missing(monkeypatch):
    """Ensure `generate_answer` reports an error when the Gemini key is missing."""
    from src.ai_pdf_assistant import answer_generation
    monkeypatch.setattr(answer_generation, "load_dotenv", lambda: None)

    def fail_openrouter(_prompt):
        raise RuntimeError("OpenRouter unavailable")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(answer_generation, "_generate_with_openrouter", fail_openrouter)

    passages = [
        {"text": "Virtual memory allows processes to use an address space larger than physical memory.", "page_number": 3}
    ]
    result = generate_answer("What is virtual memory?", passages)

    assert "local extractive fallback" not in result
    assert "No available provider could generate an answer" in result


def test_generate_answer_reports_error_when_no_api_keys_are_configured(monkeypatch):
    """Ensure `generate_answer` reports an error when no API keys are configured."""
    from src.ai_pdf_assistant import answer_generation
    monkeypatch.setattr(answer_generation, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    passages = [
        {"text": "Virtual memory allows processes to use an address space larger than physical memory.", "page_number": 3}
    ]
    result = generate_answer("What is virtual memory?", passages)

    assert "local extractive fallback" not in result
    assert "No model API key is configured" in result