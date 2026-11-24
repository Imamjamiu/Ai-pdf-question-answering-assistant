
import base64
import json
import os
import re

import httpx
from dotenv import load_dotenv
from google.genai import types

from .answer_generation import load_gemini_client, GEMINI_MODEL


OPENROUTER_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
DEFAULT_VISION_MODEL = "openrouter/free"


def _build_visual_prompt(caption, page_number, visual_id):
    """Build instructions for grounded visual analysis."""

    return f"""
You are analysing a visual extracted from an academic PDF.

VISUAL METADATA:
- Figure or caption: {caption or "No caption provided"}
- PDF page: {page_number}
- Visual ID: {visual_id or "Not provided"}

TASK:
Carefully inspect the supplied image and explain its contents.

Identify:
1. The type of visual: diagram, chart, table, graph,
   illustration, flowchart or other.
2. Its main purpose and what it communicates.
3. Important visible labels, components and values.
4. Relationships, connections, sequences or comparisons
   shown between its components.
5. Important observations directly supported by the image.

GROUNDING RULES:
- Use only information visibly supported by the image.
- Do not invent missing labels, values or relationships.
- Preserve important technical terminology and numerical values.
- If a label is unclear, identify it as unreadable or uncertain.
- Distinguish direct observations from interpretations.
- Do not assume information based only on the caption.
- If the image is incomplete or difficult to interpret,
  explicitly explain the limitation.

Return ONLY valid JSON using this structure:

{{
    "visual_type": "type of visual",
    "title": "short descriptive title",
    "purpose": "what the visual communicates",
    "summary": "clear explanation of the visual",
    "key_elements": [
        "important visible element"
    ],
    "relationships": [
        "relationship or connection shown"
    ],
    "data_points": [
        "important visible values or measurements"
    ],
    "uncertainties": [
        "unclear or unreadable information"
    ]
}}
""".strip()


def _extract_json_response(response_text):
    """Parse JSON returned by either model provider."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("The model returned an empty response.")

    response_text = response_text.strip()

    # Handle responses wrapped in Markdown code fences.
    response_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response_text,
        flags=re.IGNORECASE
    ).strip()

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The model returned invalid JSON."
        ) from error

    if not isinstance(result, dict):
        raise ValueError("Visual analysis must be a JSON object.")

    required_fields = {
        "visual_type",
        "title",
        "purpose",
        "summary",
        "key_elements",
        "relationships",
        "data_points",
        "uncertainties"
    }

    missing_fields = required_fields.difference(result)

    if missing_fields:
        raise ValueError(
            f"Visual analysis is missing fields: {sorted(missing_fields)}"
        )

    text_fields = (
        "visual_type",
        "title",
        "purpose",
        "summary"
    )

    for field in text_fields:
        if not isinstance(result[field], str):
            raise ValueError(f"Invalid text field: {field}")

    list_fields = (
        "key_elements",
        "relationships",
        "data_points",
        "uncertainties"
    )

    for field in list_fields:
        if not isinstance(result[field], list):
            raise ValueError(f"Invalid list field: {field}")

        if not all(isinstance(item, str) for item in result[field]):
            raise ValueError(f"Invalid entries in field: {field}")

    return result


def _extract_openrouter_text(payload):
    """Extract generated text from an OpenRouter Responses API result."""

    if payload.get("output_text"):
        return payload["output_text"].strip()

    text_parts = []

    for output_item in payload.get("output", []):
        for content_item in output_item.get("content", []):
            if content_item.get("type") in ("output_text", "text"):
                text = content_item.get("text", "")

                if text:
                    text_parts.append(text)

    result = "\n".join(text_parts).strip()

    if not result:
        raise ValueError(
            "OpenRouter returned no usable visual analysis text."
        )

    return result


def _analyze_with_openrouter(prompt, image_bytes):
    """Analyse an image using OpenRouter's multimodal Responses API."""

    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    model = os.getenv(
        "OPENROUTER_VISION_MODEL",
        DEFAULT_VISION_MODEL
    ).strip()

    if not model:
        model = DEFAULT_VISION_MODEL

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = f"data:image/png;base64,{encoded_image}"

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "auto"
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = httpx.post(
        OPENROUTER_RESPONSES_URL,
        headers=headers,
        json=payload,
        timeout=90.0
    )

    if not response.is_success:
        raise RuntimeError(
            f"OpenRouter returned HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    return _extract_openrouter_text(response.json())


def _analyze_with_gemini(prompt, image_bytes):
    """Analyse an image using the existing Gemini client."""

    client = load_gemini_client()

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json"
        )
    )

    return (response.text or "").strip()


def analyze_visual_asset(
    image_bytes,
    caption,
    page_number,
    visual_id=None
):
    """
    Analyse an extracted PDF visual.

    Provider order:
    1. OpenRouter
    2. Gemini fallback

    Returns structured visual metadata.
    """

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("A valid image in PNG bytes is required.")

    if not isinstance(caption, str):
        caption = ""

    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError("Page number must be a positive integer.")

    prompt = _build_visual_prompt(
        caption=caption,
        page_number=page_number,
        visual_id=visual_id
    )

    provider_errors = []
    from . import answer_generation

    # 1. Try OpenRouter first.
    load_dotenv()

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if openrouter_key and not answer_generation._OPENROUTER_DAILY_QUOTA_EXHAUSTED:
        try:
            print("Visual analysis: trying OpenRouter...")

            response_text = _analyze_with_openrouter(
                prompt,
                image_bytes
            )

            analysis = _extract_json_response(response_text)

            analysis["provider"] = "openrouter"

            print("Visual analysis completed using OpenRouter.")

            return _attach_metadata(
                analysis,
                caption,
                page_number,
                visual_id
            )

        except Exception as error:
            if answer_generation._is_openrouter_daily_quota_error(error):
                answer_generation._OPENROUTER_DAILY_QUOTA_EXHAUSTED = True
                print("OpenRouter daily free-model quota exhausted.")

            provider_errors.append(
                f"OpenRouter: {error}"
            )

            print(f"OpenRouter visual analysis failed: {error}")

    else:
        if answer_generation._OPENROUTER_DAILY_QUOTA_EXHAUSTED:
            provider_errors.append(
                "OpenRouter: daily free-model quota already exhausted."
            )
        else:
            provider_errors.append(
                "OpenRouter: API key is not configured."
            )

    # ---------------------------------------------
    # 2. Try Gemini as fallback.
    # ---------------------------------------------

    try:
        if answer_generation._GEMINI_DAILY_QUOTA_EXHAUSTED:
            raise RuntimeError(
                "Gemini daily request quota already exhausted."
            )

        print("Visual analysis: trying Gemini fallback...")

        response_text = _analyze_with_gemini(
            prompt,
            image_bytes
        )

        analysis = _extract_json_response(response_text)

        analysis["provider"] = "gemini"

        print("Visual analysis completed using Gemini.")

        return _attach_metadata(
            analysis,
            caption,
            page_number,
            visual_id
        )

    except Exception as error:
        if answer_generation._is_gemini_daily_quota_error(error):
            answer_generation._GEMINI_DAILY_QUOTA_EXHAUSTED = True
            print("Gemini daily request quota exhausted.")

        provider_errors.append(
            f"Gemini: {error}"
        )

        print(f"Gemini visual analysis failed: {error}")

    # ---------------------------------------------
    # 3. Both providers failed.
    # Preserve the extracted visual and continue processing.
    # ---------------------------------------------

    print(
        "Visual analysis unavailable. "
        "The original extracted visual will still be preserved."
    )

    analysis = {
        "visual_type": "not_analyzed",
        "title": caption or "PDF visual",
        "purpose": "",
        "summary": (
            "AI visual analysis is temporarily unavailable. "
            "The original figure was extracted, but its contents "
            "have not been verified by a vision model."
        ),
        "key_elements": [],
        "relationships": [],
        "data_points": [],
        "uncertainties": [
            "The figure has not been analysed by a vision model."
        ],
        "analysis_status": "unavailable",
        "provider": "unavailable",
    }

    return _attach_metadata(
        analysis,
        caption,
        page_number,
        visual_id,
    )
def _attach_metadata(analysis, caption, page_number, visual_id):
    """Attach original PDF metadata to the analysis."""

    analysis["visual_id"] = visual_id
    analysis["caption"] = caption
    analysis["page_number"] = page_number

    return analysis
