import hashlib
import re

import streamlit as st

from ai_pdf_assistant.pdf_processing import extract_pdf_pages

from ai_pdf_assistant.retrieval import (
    split_pages_into_chunks,
    create_embeddings,
    create_keyword_index,
    find_relevant_passages,
    add_neighboring_chunks
)

from ai_pdf_assistant.answer_generation import generate_answer

def get_cited_pages(answer, context_passages):
    """Return cited PDF pages that have matching retrieved passages."""
    available_pages = {
        str(result.get("page_number"))
        for result in context_passages
        if result.get("page_number") is not None
    }

    cited_pages = []
    for page_number in re.findall(
        r"\[PDF page\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE,
    ):
        if page_number in available_pages and page_number not in cited_pages:
            cited_pages.append(page_number)

    return cited_pages


def remove_sources_line(answer):
    """Hide the generated Sources line because page controls replace it."""
    return re.sub(
        r"(?im)^\s*Sources\s*:.*$",
        "",
        answer,
    ).strip()


def clean_answer_for_display(answer):
    """Hide inline citation tokens because compact page controls replace them."""
    answer = remove_sources_line(answer)
    answer = re.sub(
        r"\[PDF page\s+\d+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"(?<!\*)\*{3,}(?!\*)", "", answer)
    answer = re.sub(r"(?<=\w) {2,}(?=\w)", " ", answer)
    answer = re.sub(r"[ \t]+([,.;:!?])", r"\1", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()



def is_summary_follow_up(question):
    """Recognize requests to summarize the previous answer."""
    return bool(re.match(
        r"^\s*(?:please\s+)?(?:summari[sz]e|shorten)\s+"
        r"(?:(?:your|that|this|the previous|the last)\s+)?"
        r"(?:answer|response|explanation|it|that|this)\s*[?.!]*$",
        question,
        flags=re.IGNORECASE,
    ))


def is_contextual_follow_up(question, active_topic):
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
    if not any(
        re.search(pattern, question, flags=re.IGNORECASE)
        for pattern in explicit_follow_up_patterns
    ):
        return False

    # A named subject makes a question searchable on its own, even if it
    # contains a pronoun such as "it" later in the sentence.
    generic_words = {
        "a", "about", "also", "an", "and", "are", "as", "can", "could",
        "did", "does", "do", "explain", "further", "give", "how", "i",
        "is", "it", "me", "more", "of", "please", "that", "the", "their",
        "them", "then", "this", "to", "what", "when", "where", "which",
        "who", "why", "with", "would", "you",
    }
    content_words = [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", question)
        if word.lower() not in generic_words
    ]
    return len(content_words) <= 3


def build_retrieval_query(question, active_topic=None):
    """Add only the current topic when a question is an incomplete follow-up."""
    if not active_topic:
        return question

    return (
        f"Recent topic: {active_topic[:400]}\n"
        f"Current follow-up question: {question}"
    )

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
# 1. Configure the application
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Question-Answering Assistant")

st.write(
    "Upload a PDF and ask questions about it. Answers use that PDF as "
    "their source; if it doesn't contain the answer, the app will say so."
)

st.markdown(
    "**How to use it**\n"
    "1. Upload a PDF with selectable text.\n"
    "2. Ask one or more questions about the PDF.\n"
    "3. Ask follow-ups or start a new question about the PDF.\n"
    "4. Type summarize your answer to summarize the previous answer.\n"
    "5. Click a citation to view its passage, or use the answer buttons for more help."
)
# --------------------------------------------------
# 2. Process the uploaded PDF
# --------------------------------------------------

@st.cache_data(show_spinner=False)

def process_uploaded_pdf(pdf_bytes):

    # Extract individual PDF pages
    page_count, pages = extract_pdf_pages(pdf_bytes)

    # Create one text string for document statistics
    full_text = "\n".join(
        page["text"] for page in pages
    )

    # Create chunks while preserving page numbers
    chunks = split_pages_into_chunks(pages)

    # Generate semantic embeddings
    embeddings = create_embeddings(chunks)

    # Create keyword index
    keyword_index = create_keyword_index(chunks)

    return (
        page_count,
        full_text,
        chunks,
        embeddings,
        keyword_index
    )

# --------------------------------------------------
# 3. Upload a PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)


if uploaded_file is not None:
    try:

        with st.spinner("Processing your PDF..."):

            pdf_bytes = uploaded_file.getvalue()

            page_count, full_text, chunks, embeddings, keyword_index = (
                process_uploaded_pdf(pdf_bytes)
            )

        if not full_text.strip():

            st.warning(
                "No readable text was found. This app needs a text-based PDF; "
                "scanned pages are not processed with OCR."
            )

        else:

            st.success("PDF processed successfully!")

            # Display document information
            col1, col2, col3 = st.columns(3)

            col1.metric("PDF pages", page_count)

            col2.metric("Text chunks", len(chunks))

            col3.metric(
                "Extracted characters",
                f"{len(full_text):,}"
            )

            st.divider()

            # ------------------------------------------
            # 4. Chat with this PDF
            # ------------------------------------------

            document_id = hashlib.sha256(pdf_bytes).hexdigest()
            if (
                st.session_state.get("pdf_chat_document_id")
                != document_id
            ):
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

            st.subheader("Ask questions about this PDF")
            if st.session_state["pdf_chat_messages"]:
                if st.button("Clear conversation"):
                    st.session_state["pdf_chat_messages"] = []
                    st.session_state["pdf_chat_active_topic"] = None
                    st.session_state["pending_answer_action"] = None

            chat_history_area = st.container()
            question = st.chat_input(
                "Ask a question about the uploaded PDF..."
            )

            messages = st.session_state["pdf_chat_messages"]
            if question:
                prior_history = list(messages)
                summary_request = is_summary_follow_up(question)

                if summary_request:
                    previous_answer = next(
                        (
                            item
                            for item in reversed(prior_history)
                            if item.get("role") == "assistant"
                        ),
                        None,
                    )
                    context_passages = (
                        previous_answer.get("passages", [])
                        if previous_answer
                        else []
                    )
                    if context_passages:
                        with st.spinner("Summarizing the previous answer..."):
                            answer = generate_answer(
                                ANSWER_ACTIONS["summary"]["prompt"],
                                context_passages,
                                conversation_history=prior_history[-2:],
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
                    active_topic = st.session_state.get(
                        "pdf_chat_active_topic"
                    )
                    follow_up = is_contextual_follow_up(
                        question,
                        active_topic,
                    )
                    if not follow_up:
                        active_topic = question.strip()
                        st.session_state["pdf_chat_active_topic"] = active_topic

                    retrieval_query = build_retrieval_query(
                        question,
                        active_topic if follow_up else None,
                    )

                    with st.spinner(
                        "Searching this PDF and generating an answer..."
                    ):
                        relevant_passages = find_relevant_passages(
                            question=retrieval_query,
                            chunks=chunks,
                            embeddings=embeddings,
                            top_k=8,
                            keyword_index=keyword_index,
                        )

                        if relevant_passages:
                            context_passages = add_neighboring_chunks(
                                relevant_passages,
                                chunks,
                            )
                            answer = generate_answer(
                                question,
                                context_passages,
                                conversation_history=(
                                    prior_history[-4:] if follow_up else None
                                ),
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
                    },
                ])
            pending_action = st.session_state.get("pending_answer_action")
            if pending_action:
                st.session_state["pending_answer_action"] = None
                target_index = pending_action.get("message_index")
                action_key = pending_action.get("action_key")
                action = ANSWER_ACTIONS.get(action_key)

                if (
                    isinstance(target_index, int)
                    and 0 <= target_index < len(messages)
                    and messages[target_index].get("role") == "assistant"
                    and action
                ):
                    target_message = messages[target_index]
                    action_passages = target_message.get("passages", [])
                    if action_passages:
                        with st.spinner(f'{action["label"]}...'):
                            action_answer = generate_answer(
                                action["prompt"],
                                action_passages,
                                conversation_history=messages[
                                    :target_index + 1
                                ][-4:],
                            )
                        messages.extend([
                            {
                                "role": "user",
                                "content": f'{action["label"]}.',
                            },
                            {
                                "role": "assistant",
                                "content": action_answer,
                                "passages": action_passages,
                            },
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
                        cited_pages = get_cited_pages(
                            answer_content,
                            context_passages,
                        )

                        if display_answer:
                            st.markdown(display_answer)

                        if cited_pages:
                            st.caption(
                                "Click a cited page to show its evidence; "
                                "click it again to hide it."
                            )
                            for page_start in range(0, len(cited_pages), 4):
                                page_row = cited_pages[
                                    page_start:page_start + 4
                                ]
                                columns = st.columns(
                                    [1] * len(page_row)
                                    + [max(1, 8 - len(page_row))],
                                    gap="small",
                                )
                                for page_number, column in zip(
                                    page_row,
                                    columns,
                                ):
                                    with column:
                                        with st.popover(
                                            f"[PDF page {page_number}]"
                                        ):
                                            matching_passages = [
                                                result
                                                for result in context_passages
                                                if str(result.get("page_number"))
                                                == page_number
                                            ]
                                            st.markdown(
                                                f"**Supporting text from PDF "
                                                f"page {page_number}**"
                                            )
                                            for result in matching_passages:
                                                st.write(result["text"])
                        if context_passages:
                            action_columns = st.columns(4, gap="small")
                            for action_key, action_column in zip(
                                ANSWER_ACTIONS,
                                action_columns,
                            ):
                                action = ANSWER_ACTIONS[action_key]
                                with action_column:
                                    if st.button(
                                        action["label"],
                                        key=(
                                            f"answer-action-{message_index}-"
                                            f"{action_key}"
                                        ),
                                        help=action["help"],
                                    ):
                                        st.session_state[
                                            "pending_answer_action"
                                        ] = {
                                            "message_index": message_index,
                                            "action_key": action_key,
                                        }
                                        st.rerun()


    except Exception as error:

        print(f"PDF processing error: {error!r}")
        st.error(
            "I couldn't finish processing this PDF. Check the PyCharm Run "
            "console for technical details, then try again."
        )
