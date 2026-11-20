# AI PDF Assistant

A Streamlit app that lets you upload a text-based PDF, ask follow-up questions about it, and review the passages used to generate each answer.

## What it does

1. Extracts text page by page from an uploaded PDF.
2. Splits the text into chunks while keeping each PDF page number.
3. Searches with sentence-transformer embeddings and TF-IDF keyword matching, then combines the rankings.
4. Sends the strongest passages, with nearby context, to a language model.
5. Shows the complete answer first, followed by compact page citations; click a citation to open its supporting passage in a popover.
6. Keeps an in-session conversation for the current PDF, with context-aware follow-ups and independent new questions. Answer buttons offer clearer explanations, more detail, summaries, and examples. Short follow-ups use the current topic, complete questions start fresh searches, and "summarize your answer" reuses the previous answer's passages. Answers still need support from the uploaded PDF; changing the PDF clears the conversation.

## OCR setup

OCR is performed locally with the open-source Tesseract engine; the app does not send PDF pages to a paid OCR API.

- **Windows / PyCharm:** install the Python dependencies with `python -m pip install -r requirements.txt`. Install the Tesseract OCR application separately, add the folder containing `tesseract.exe` to your Windows PATH, restart the terminal, and verify with `tesseract --version`.
- **Streamlit Community Cloud:** the root-level `packages.txt` installs the `tesseract-ocr` system package during deployment.
- The default OCR language is English (`eng`). Other languages require their Tesseract language data to be installed.

Pages with little selectable text are OCRed as a whole. On mixed pages, meaningful embedded-image regions are OCRed separately and merged with the native text. OCR extracts text; it does not yet understand diagram structure or display the original figures and tables.

## Requirements

- Python 3.10 or newer
- Streamlit 1.32 or newer (required for clickable citation popovers)
- An API key for OpenRouter or Gemini to generate answers
- Internet access for the configured answer provider. The embedding model may also need to be downloaded the first time it is used.

## Run in PyCharm on Windows

1. Open the project folder in PyCharm.
2. Select the Python environment you want to use for the project.
3. In PyCharm's terminal, install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add at least one provider key:

   ```text
   OPENROUTER_API_KEY=your_openrouter_key
   GEMINI_API_KEY=your_gemini_key
   ```

   You can configure either key or both. If both are set, the app tries OpenRouter first and uses Gemini as a fallback. `OPENROUTER_MODEL` is optional; it defaults to `openrouter/free`.

5. Start the app from the project terminal:

   ```powershell
   streamlit run main.py
   ```

   Open the local address printed in the terminal, upload a PDF, and ask questions. You can ask follow-ups in the same conversation; each answer is grounded in retrieved passages from that PDF.

The `.gitignore` excludes `.env`, so provider keys should stay out of commits. Never paste a real API key into source code or share it in screenshots.

## Evaluation

The project includes a 30-question evaluation set in `evaluation/questions.json`. Each entry contains a question, expected source page numbers, and a short answer key.

Run the local retrieval check with:

```powershell
python evaluate_retrieval.py
```

This checks expected-page coverage in the same context used by the app: the top eight retrieved chunks plus neighboring chunks around the top two results. It reports context hit rate, expected-page recall in that context, and mean reciprocal rank for the ranked top eight. It does **not** call Gemini or OpenRouter and does not grade generated answer quality. The local sentence-transformer model must be available for retrieval.

To review generated answers without using API quota, run:

```powershell
python evaluate_answers.py
```

The first run creates `evaluation/answer_review.csv` with the 30 questions and answer keys. After checking an answer against the displayed passages, fill in `answer_status` and `citation_status` with `pass`, `partial`, or `fail`, then run the command again for a summary. Mark the answer `pass` when it is correct and supported by the passages; mark citations `pass` when their page numbers point to the evidence shown. Use `partial` for an answer or citation with a minor problem, and `fail` for a major error. Use `review_notes` to explain partial or failed ratings. This is a manual review: the script counts your ratings and does not judge answers automatically or save generated answers.

Run the focused retrieval unit tests with:

```powershell
python -m pytest tests/test_retrieval_unit.py -v
```

These tests check retrieval helpers; they do not verify the quality or factual accuracy of live model answers.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Streamlit PDF chat interface and page-linked citations |
| `pdf_processing.py` | PDF text extraction |
| `retrieval.py` | Page-aware chunking, embeddings, keyword search, and hybrid ranking |
| `answer_generation.py` | Prompt construction, OpenRouter/Gemini requests, and citation formatting |
| `evaluation/questions.json` | Questions, expected pages, and answer keys |
| `evaluation/answer_review.csv` | Manual answer and citation review sheet |
| `evaluate_retrieval.py` | Offline retrieval evaluation |
| `evaluate_answers.py` | Summary report for manual answer reviews |
| `tests/test_retrieval_unit.py` | Focused retrieval unit tests |

## Current limitations

- Scanned pages and text inside meaningful embedded-image regions are processed with local Tesseract OCR. OCR extracts text; it does not yet interpret diagrams or preserve figures as retrievable visual assets.
- Retrieval can miss a useful passage, and generated answers or citations can still be incomplete or incorrect. Check the supporting passages before relying on an answer.
- Answer generation depends on provider availability, account limits, and API quotas.
- The bundled evaluation set covers one virtual-memory PDF. More documents and questions are needed to assess performance across different PDFs.
