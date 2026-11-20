import hashlib
import streamlit as st
from ai_pdf_assistant.pdf_processing import extract_pdf_pages, OcrDependencyError
from ai_pdf_assistant.retrieval import (
    split_pages_into_chunks, create_embeddings, create_keyword_index,
    find_relevant_passages, add_neighboring_chunks,
)
from ai_pdf_assistant.answer_generation import generate_answer
from ai_pdf_assistant.conversation import (
    is_acknowledgement, is_summary_follow_up, is_contextual_follow_up,
    build_retrieval_query, get_cited_pages, clean_answer_for_display,
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

# Configure the application
st.set_page_config(page_title="AI PDF Assistant", page_icon="📚", layout="wide")
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

# Process the uploaded PDF
@st.cache_data(show_spinner=False)
def process_uploaded_pdf(pdf_bytes):
    # Extract individual PDF pages
    page_count, pages = extract_pdf_pages(pdf_bytes)
    # Create one text string for document statistics
    full_text = "\n".join(page["text"] for page in pages)
    # Create chunks while preserving page numbers
    chunks = split_pages_into_chunks(pages)
    # Generate semantic embeddings
    embeddings = create_embeddings(chunks)
    # Create keyword index
    keyword_index = create_keyword_index(chunks)
    return page_count, full_text, chunks, embeddings, keyword_index

# Upload a PDF
uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])
if uploaded_file:
    try:
        with st.spinner("Processing your PDF..."):
            pdf_bytes = uploaded_file.getvalue()
            page_count, full_text, extracted_pages, chunks, embeddings, keyword_index = process_uploaded_pdf(pdf_bytes)

        if not full_text.strip():
            st.warning(
                "No readable text was found. This app needs a text-based PDF; "
                "scanned pages are not processed with OCR."
            )
        else:
            st.success("PDF processed successfully!")
            native_pages = sum(page.get("extraction_method") == "native" for page in extracted_pages)
            ocr_pages = sum(page.get("extraction_method") == "ocr" for page in extracted_pages)
            mixed_pages = sum(page.get("extraction_method") == "mixed" for page in extracted_pages)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("PDF pages", page_count)
            col2.metric("Text chunks", len(chunks))
            col3.metric("OCR-processed pages", ocr_pages + mixed_pages)
            col4.metric("Extracted characters", f"{len(full_text):,}")
            with st.expander("Text extraction details"):
                st.write(f"Pages with native text: {native_pages}")
                st.write(f"Pages processed with OCR: {ocr_pages}")
                st.write(f"Pages containing both native and OCR text: {mixed_pages}")
                st.caption("OCR runs locally through Tesseract. Embedded images on pages that also contain selectable text are OCRed separately.")
            # Display document information
            col1, col2, col3 = st.columns(3)
            col1.metric("PDF pages", page_count)
            col2.metric("Text chunks", len(chunks))
            col3.metric("Extracted characters", f"{len(full_text):,}")
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

            st.subheader("Ask questions about this PDF")
            if st.session_state["pdf_chat_messages"]:
                if st.button("Clear conversation"):
                    st.session_state["pdf_chat_messages"] = []
                    st.session_state["pdf_chat_active_topic"] = None
                    st.session_state["pending_answer_action"] = None

            chat_history_area = st.container()
            question = st.chat_input("Ask a question about the uploaded PDF...")
            messages = st.session_state["pdf_chat_messages"]

            if question and is_acknowledgement(question):
                messages.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "You're welcome! Let me know if you have another question.", "passages": []}
                ])
                question = None

            if question:
                prior_history = list(messages)
                summary_request = is_summary_follow_up(question)

                if summary_request:
                    previous_answer = next(
                        (item for item in reversed(prior_history) if item.get("role") == "assistant"), None
                    )
                    context_passages = previous_answer.get("passages", []) if previous_answer else []
                    if context_passages:
                        with st.spinner("Summarizing the previous answer..."):
                            answer = generate_answer(
                                ANSWER_ACTIONS["summary"]["prompt"], context_passages, conversation_history=prior_history[-2:]
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
                            question=retrieval_query, chunks=chunks, embeddings=embeddings, top_k=8, keyword_index=keyword_index
                        )
                        if relevant_passages:
                            context_passages = add_neighboring_chunks(relevant_passages, chunks)
                            answer = generate_answer(
                                question, context_passages, conversation_history=(prior_history[-4:] if follow_up else None)
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
                    {"role": "assistant", "content": answer, "passages": context_passages},
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
                                action["prompt"], action_passages, conversation_history=messages[:target_index + 1][-4:]
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

                        with st.container(gap="small"):
                            if cited_pages:
                                with st.container(
                                    horizontal=True, horizontal_alignment="left", gap="small", vertical_alignment="center"
                                ):
                                    st.markdown(
                                        "<span style='font-size: 0.875rem; opacity: 0.6;'>Click a cited page to view evidence:</span>",
                                        unsafe_allow_html=True, width="content",
                                    )
                                    for page_number in cited_pages:
                                        with st.popover(f"[page {page_number}]"):
                                            matching_passages = [
                                                result for result in context_passages
                                                if str(result.get("page_number")) == page_number
                                            ]
                                            st.markdown(f"**Supporting text from PDF page {page_number}**")
                                            for result in matching_passages:
                                                st.write(result["text"])

                            if context_passages:
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
    except OcrDependencyError as error:
        st.error(str(error))
    except Exception as error:
        print(f"PDF processing error: {error!r}")
        st.error(
            "I couldn't finish processing this PDF. "
            "Possible reasons: the file is corrupted, password-protected, "
            "or contains only scanned images (OCR is not supported yet). "
            "Check the console for technical details."
        )