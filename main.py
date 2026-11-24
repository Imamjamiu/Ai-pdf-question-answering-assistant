import hashlib
import re
import streamlit as st
from ai_pdf_assistant.pdf_processing import extract_pdf_pages
from ai_pdf_assistant.retrieval import (
    split_pages_into_chunks, create_embeddings, create_keyword_index,
    find_relevant_passages, add_neighboring_chunks,
)
from ai_pdf_assistant.answer_generation import generate_answer
from ai_pdf_assistant.conversation import (
    is_acknowledgement, is_summary_follow_up, is_contextual_follow_up,
    build_retrieval_query, get_cited_pages, clean_answer_for_display,
)

from ai_pdf_assistant.visual_processing import (
    extract_visual_assets,
    extract_figure_captions,
    detect_figure_regions,
    render_page_region,
)

from ai_pdf_assistant.visual_understanding import analyze_visual_asset


ANSWER_ACTIONS = {
    "clearer": {
        "label": "Explain more clearly",
        "prompt": (
            "Re-explain the previous answer in simpler, clearer language. "
            "Use short sentences, define technical terms, and preserve the "
            "important points. Use only the supplied PDF passages and "
            "include citations."
        ),
        "help": "Rewrite the answer in easier language.",
    },
    "more": {
        "label": "Explain more",
        "prompt": (
            "Explain the previous answer in more detail. Clarify how its "
            "important points connect. Use only the supplied PDF passages "
            "and include citations."
        ),
        "help": "Add useful detail from the PDF.",
    },
    "summary": {
        "label": "Summarize",
        "prompt": (
            "Summarize the previous answer in 3 to 5 concise bullet points. "
            "Keep its key ideas, use only the supplied PDF passages, and "
            "include citations."
        ),
        "help": "Make the answer shorter.",
    },
    "example": {
        "label": "Give an example",
        "prompt": (
            "Give one short example that helps explain the previous answer. "
            "Use only concepts supported by the supplied PDF passages. If "
            "the passages do not support a useful example, say so instead "
            "of inventing facts. Include citations."
        ),
        "help": "Illustrate the answer using the PDF.",
    },
}

# --------------------------------------------------
# Professional application interface
# -------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI | PDF Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --dm-accent: #635BFF;
        --dm-ink: var(--text-color, #F4F6FB);
        --dm-muted: color-mix(in srgb, var(--text-color, #F4F6FB) 68%, transparent);
        --dm-line: rgba(127, 140, 160, .22);
        --dm-surface: rgba(127, 140, 160, .055);
    }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 5rem; }
    [data-testid="stHeader"] { background: transparent; }
    .dm-brand { display:flex; align-items:center; gap:.65rem; color:var(--dm-ink);
        font-weight:750; letter-spacing:-.025em; font-size:1.08rem; margin-bottom:2rem; }
    .dm-brand-mark { width:2.35rem; height:2.35rem; display:grid; place-items:center;
        border-radius:.8rem; color:white; font-size:1.15rem;
        background:linear-gradient(135deg,#766DFF,#5148E5);
        box-shadow:0 7px 18px rgba(99,91,255,.22); }
    .dm-eyebrow { color:var(--dm-accent); text-transform:uppercase; letter-spacing:.13em;
        font-size:.72rem; font-weight:750; margin-bottom:.55rem; }
    .dm-title { color:var(--dm-ink); font-size:clamp(2rem,4vw,3.05rem);
        line-height:1.08; letter-spacing:-.055em; font-weight:780; margin:0 0 .75rem 0; }
    .dm-subtitle { color:var(--dm-muted); font-size:1.02rem; line-height:1.65;
        max-width:720px; margin-bottom:1.7rem; }
    .dm-success { display:flex; align-items:center; gap:.8rem;
        border:1px solid rgba(22,163,74,.28); background:rgba(22,163,74,.07);
        border-radius:1rem; padding:1rem 1.15rem; margin:.75rem 0 1rem 0; }
    .dm-success-icon { width:2.35rem; height:2.35rem; display:grid; place-items:center;
        border-radius:50%; color:#15803d; background:rgba(22,163,74,.13);
        font-size:1.2rem; flex:0 0 auto; }
    .dm-success-title { font-weight:700; color:var(--dm-ink); }
    .dm-success-meta { color:var(--dm-muted); font-size:.88rem; margin-top:.15rem; }
    .dm-section-title { color:var(--dm-ink); font-size:1.18rem; font-weight:720;
        letter-spacing:-.02em; margin:.7rem 0 .3rem 0; }
    .dm-footnote { color:var(--dm-muted); font-size:.84rem; }
    div[data-testid="stChatMessage"] { border:1px solid var(--dm-line);
        border-radius:1rem; padding:1rem 1.1rem; margin-bottom:.85rem;
        background:var(--dm-surface); }
    div[data-testid="stChatInput"] { border-radius:1rem; }
    div.stButton > button { border-radius:.75rem; font-weight:620;
        border:1px solid var(--dm-line); transition:all .15s ease; }
    div.stButton > button:hover { border-color:var(--dm-accent); color:var(--dm-accent); }
    [data-testid="stFileUploader"] { border:1px dashed rgba(99,91,255,.45);
        border-radius:1rem; padding:.65rem; background:rgba(99,91,255,.035); }
    @media (max-width:700px) { .block-container { padding-left:1rem; padding-right:1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="dm-brand"><span class="dm-brand-mark">✦</span><span>DocuMind AI</span></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="dm-eyebrow">Your intelligent PDF workspace</div>', unsafe_allow_html=True)
st.markdown('<h1 class="dm-title">Understand more.<br>Search less.</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="dm-subtitle">Upload a document, ask questions in natural language, '
    'and explore clear answers grounded in your PDF.</div>',
    unsafe_allow_html=True,
)

# Process the uploaded PDF
@st.cache_data(show_spinner=False)
def process_uploaded_pdf(pdf_bytes):
    # 1. Extract individual PDF pages and OCR text.
    page_count, pages = extract_pdf_pages(pdf_bytes)

    # 2. Create one text string for document statistics.
    full_text = "\n".join(page["text"] for page in pages)

    # 3. Create text chunks while preserving page numbers.
    chunks = split_pages_into_chunks(pages)

    # 4. Generate semantic embeddings.
    embeddings = create_embeddings(chunks)

    # 5. Create the keyword index.
    keyword_index = create_keyword_index(chunks)

    # 6. Extract visual asset metadata.
    visual_assets = extract_visual_assets(pdf_bytes)

    # 7. Identify figure captions.
    figure_captions = extract_figure_captions(pdf_bytes)

    # 8. Detect figure regions using their captions.
    figure_regions = detect_figure_regions(
        pdf_bytes,
        captions=figure_captions,
    )

    return (
        page_count,
        full_text,
        chunks,
        embeddings,
        keyword_index,
        visual_assets,
        figure_captions,
        figure_regions,
    )


def retrieve_visual_evidence(
    question,
    figure_regions,
    pdf_bytes,
    context_passages=None,
    analyze_visuals=True,
):
    """
    Find and analyse figure regions relevant to the user's question.
    Explicit figure/page requests are respected. General questions
    retrieve only the strongest relevant figures.
    """
    visual_evidence = []

    if not figure_regions:
        return visual_evidence

    context_passages = context_passages or []

    # Identify explicitly requested figure number and page.
    figure_match = re.search(
        r"\bfig(?:ure)?\s*(\d+(?:\.\d+)*)\b",
        question,
        flags=re.IGNORECASE,
    )

    page_match = re.search(
        r"\bpage\s+(\d+)\b",
        question,
        flags=re.IGNORECASE,
    )

    requested_figure = figure_match.group(1) if figure_match else None
    requested_page = int(page_match.group(1)) if page_match else None

    # Words that add little value to topic matching.
    stopwords = {
        "what", "which", "where", "when", "why", "how",
        "the", "and", "for", "from", "with", "that",
        "this", "are", "was", "were", "does", "can",
        "could", "would", "should", "explain", "describe",
        "show", "tell", "give", "about", "into", "your",
        "have", "has", "its", "their", "there", "use",
        "using", "figure", "fig", "page",
    }

    def get_terms(text):
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {
            word for word in words
            if len(word) > 2 and word not in stopwords
        }

    question_terms = get_terms(question)

    # Collect retrieved PDF text by page.
    page_text = {}

    for passage in context_passages:
        page_number = passage.get("page_number")
        text = passage.get("text", "")

        if page_number is not None:
            page_text.setdefault(page_number, "")
            page_text[page_number] += " " + text

    # Explicit figure/page requests retain their exact matching behaviour.
    selected_regions = []

    if requested_figure or requested_page is not None:
        for index, region in enumerate(figure_regions, start=1):
            caption = region.get("caption", "")

            if not caption or region.get("status") == "no_visual_elements_found":
                continue

            if requested_figure:
                caption_match = re.search(
                    r"\bfig(?:ure)?\s*(\d+(?:\.\d+)*)\b",
                    caption,
                    flags=re.IGNORECASE,
                )

                if not caption_match:
                    continue

                if caption_match.group(1) != requested_figure:
                    continue

            if (
                requested_page is not None
                and region.get("page_number") != requested_page
            ):
                continue

            selected_regions.append((index, region))

    else:
        # General question: rank figures by caption and page relevance.
        scored_regions = []

        for index, region in enumerate(figure_regions, start=1):
            caption = region.get("caption", "")

            if not caption or region.get("status") == "no_visual_elements_found":
                continue

            page_number = region.get("page_number")
            caption_terms = get_terms(caption)
            page_terms = get_terms(page_text.get(page_number, ""))

            caption_overlap = len(question_terms & caption_terms)
            page_overlap = len(question_terms & page_terms)

            # Caption matches carry more weight than page-text matches.
            score = (caption_overlap * 3) + page_overlap

            if score > 0:
                scored_regions.append((score, index, region))

        # Highest relevance first; show no more than three.
        scored_regions.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected_regions = [
            (index, region)
            for score, index, region in scored_regions[:3]
        ]

    # Render and analyse only selected figures.
    for index, region in selected_regions:
        bbox = region.get("bbox")

        if not bbox:
            continue

        image_bytes = render_page_region(
            pdf_bytes,
            page_number=region["page_number"],
            bbox=bbox,
            scale=2,
        )

        visual_id = f"V{index}"

        analysis = None

        if analyze_visuals:
            analysis = analyze_visual_asset(
                image_bytes=image_bytes,
                caption=region.get("caption", ""),
                page_number=region["page_number"],
                visual_id=visual_id,
            )

        visual_evidence.append({
            "visual_id": visual_id,
            "page_number": region["page_number"],
            "caption": region.get("caption", ""),
            "image_bytes": image_bytes,
            "description": analysis,
        })

    return visual_evidence



# Upload a PDF
st.markdown('<div class="dm-section-title">Start with a document</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dm-footnote">Choose a PDF to begin a private, document-grounded conversation.</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"], label_visibility="collapsed")
if uploaded_file:
    try:
        with st.spinner("Processing your PDF..."):
            pdf_bytes = uploaded_file.getvalue()
            (
                page_count,
                full_text,
                chunks,
                embeddings,
                keyword_index,
                visual_assets,
                figure_captions,
                figure_regions,
            ) = process_uploaded_pdf(pdf_bytes)

        if not full_text.strip():
            st.warning(
                "We couldn't extract readable text from this document. "
                "Try another PDF or check that the pages are clear and readable."
            )
        else:
            st.markdown(
                '<div class="dm-success">'
                '<div class="dm-success-icon">✓</div>'
                '<div><div class="dm-success-title">PDF uploaded successfully</div>'
                f'<div class="dm-success-meta">{uploaded_file.name} · {page_count} pages · Ready to explore</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="dm-section-title">Chat with your document</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="dm-footnote">Ask a question about your PDF. '
                'Answers are grounded in the contents of your PDF.</div>',
                unsafe_allow_html=True,
            )
            st.divider()


            # Chat with this PDF
            document_id = hashlib.sha256(pdf_bytes).hexdigest()
            if st.session_state.get("pdf_chat_document_id") != document_id:
                st.session_state["pdf_chat_document_id"] = document_id
                st.session_state["pdf_chat_messages"] = []
                st.session_state["pdf_chat_active_topic"] = None
                st.session_state["pending_answer_action"] = None

            if "pdf_chat_messages" not in st.session_state:
                st.session_state["pdf_chat_messages"] = []
            if "pdf_chat_active_topic" not in st.session_state:
                st.session_state["pdf_chat_active_topic"] = None
            if "pending_answer_action" not in st.session_state:
                st.session_state["pending_answer_action"] = None

            if st.session_state["pdf_chat_messages"]:
                if st.button("↺ Clear conversation", help="Start a fresh conversation with this PDF"):
                    st.session_state["pdf_chat_messages"] = []
                    st.session_state["pdf_chat_active_topic"] = None
                    st.session_state["pending_answer_action"] = None
                    st.rerun()

            messages = st.session_state["pdf_chat_messages"]



            chat_history_area = st.container()
            question = st.chat_input("Ask anything about your PDF…")
            if not question:
                question = st.session_state.pop("suggested_question", None)

            if question and is_acknowledgement(question):
                messages.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "You're welcome! Let me know if you have another question.",
                     "passages": []}
                ])
                question = None

            if question:
                prior_history = list(messages)
                summary_request = is_summary_follow_up(question)
                visual_evidence = []

                visual_keywords = {
                    "figure",
                    "diagram",
                    "chart",
                    "table",
                    "image",
                    "illustration",
                    "visual",
                    "depicted",
                    "shown",
                }

                analyze_visuals = any(
                    keyword in question.lower()
                    for keyword in visual_keywords
                )

                if summary_request:
                    previous_answer = next(
                        (item for item in reversed(prior_history) if item.get("role") == "assistant"), None
                    )
                    context_passages = previous_answer.get("passages", []) if previous_answer else []
                    if context_passages:
                        with st.spinner("Summarizing the previous answer..."):
                            answer = generate_answer(
                                ANSWER_ACTIONS["summary"]["prompt"], context_passages,
                                conversation_history=prior_history[-2:]
                            )
                    else:
                        answer = (
                            "I don't have supporting PDF passages from the "
                            "previous answer to summarize. Ask the question "
                            "again and I can search the PDF."
                        )
                elif not chunks:
                    context_passages = []
                    answer = "No searchable text was found in this PDF."
                else:
                    active_topic = st.session_state.get("pdf_chat_active_topic")
                    follow_up = is_contextual_follow_up(question, active_topic)
                    if not follow_up:
                        active_topic = question.strip()
                        st.session_state["pdf_chat_active_topic"] = active_topic

                    retrieval_query = build_retrieval_query(question, active_topic if follow_up else None)
                    with st.spinner("Searching this PDF and generating an answer..."):
                        relevant_passages = find_relevant_passages(
                            question=retrieval_query, chunks=chunks, embeddings=embeddings, top_k=8,
                            keyword_index=keyword_index
                        )

                        if relevant_passages:
                            context_passages = add_neighboring_chunks(
                                relevant_passages,
                                chunks,
                            )

                            # Retrieve and analyse relevant original PDF figures.
                            if analyze_visuals:
                                visual_evidence = retrieve_visual_evidence(
                                    question,
                                    figure_regions,
                                    pdf_bytes,
                                    context_passages=context_passages,
                                    analyze_visuals=True,
                                )

                            # Generate an answer using both text and visual evidence.
                            answer = generate_answer(
                                question,
                                context_passages,
                                conversation_history=(
                                    prior_history[-4:] if follow_up else None
                                ),
                                visual_evidence=visual_evidence,
                            )

                        else:
                            context_passages = []
                            answer = (
                                "I couldn't find enough support for that "
                                "answer in this PDF. Try asking about a "
                                "topic covered in the document."
                            )

                messages.extend([
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": answer,
                        "passages": context_passages,
                        "visual_evidence": visual_evidence,
                    }
                ])

            pending_action = st.session_state.get("pending_answer_action")
            if pending_action:
                st.session_state["pending_answer_action"] = None
                target_index = pending_action.get("message_index")
                action_key = pending_action.get("action_key")
                action = ANSWER_ACTIONS.get(action_key)

                if (
                        isinstance(target_index, int) and 0 <= target_index < len(messages)
                        and messages[target_index].get("role") == "assistant" and action
                ):
                    target_message = messages[target_index]
                    action_passages = target_message.get("passages", [])
                    if action_passages:
                        with st.spinner(f'{action["label"]}...'):
                            action_answer = generate_answer(
                                action["prompt"], action_passages,
                                conversation_history=messages[:target_index + 1][-4:]
                            )
                        messages.extend([
                            {"role": "user", "content": f'{action["label"]}.'},
                            {"role": "assistant", "content": action_answer, "passages": action_passages},
                        ])

            with chat_history_area:
                for message_index, message in enumerate(messages):
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(message["content"])
                        continue

                    with st.chat_message("assistant"):
                        context_passages = message.get("passages", [])
                        answer_content = message["content"]
                        display_answer = clean_answer_for_display(answer_content)
                        cited_pages = get_cited_pages(answer_content, context_passages)

                    if display_answer:
                        st.markdown(display_answer)

                        # Display original PDF figures used as visual evidence.
                    message_visuals = message.get("visual_evidence", [])

                    for visual in message_visuals:
                        st.markdown(
                            f"**Figure from your PDF · Page {visual['page_number']}: "
                            f"{visual['caption']}**"
                        )

                        st.image(
                            visual["image_bytes"],
                            caption=visual["caption"],
                            use_container_width=True,
                        )

                    with st.container(gap="small"):
                        if cited_pages:
                            with st.container(
                                    horizontal=True, horizontal_alignment="left", gap="small",
                                    vertical_alignment="center"
                            ):
                                st.markdown(
                                    "**Sources** · Select a page number to inspect the supporting passage:",
                                    width="content",
                                )
                                for page_number in cited_pages:
                                    with st.popover(f"[page {page_number}]"):
                                        matching_passages = [
                                            result for result in context_passages
                                            if str(result.get("page_number")) == page_number
                                        ]
                                        st.markdown(f"**Supporting passage · Page {page_number}**")
                                        for result in matching_passages:
                                            st.write(result["text"])

                        if (
                                context_passages
                                and "OpenRouter and Gemini are both unavailable right now"
                                not in answer_content
                        ):
                            with st.container(
                                horizontal=True, horizontal_alignment="left", gap=None, vertical_alignment="center",
                            ):
                                for action_key in ANSWER_ACTIONS:
                                    action = ANSWER_ACTIONS[action_key]
                                    if st.button(
                                        action["label"],
                                        key=f"answer-action-{message_index}-{action_key}",
                                        help=action["help"],
                                    ):
                                        st.session_state["pending_answer_action"] = {
                                            "message_index": message_index, "action_key": action_key,
                                        }
                                        st.rerun()
    except Exception as error:
        print(f"PDF processing error: {error!r}")
        st.error(
            "We couldn't process this PDF. Please check that the file opens "
            "normally, is not password-protected, and try uploading it again."
        )