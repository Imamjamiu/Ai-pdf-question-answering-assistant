from pathlib import Path

from ai_pdf_assistant.pdf_processing import extract_pdf_pages


PDF_PATH = Path(__file__).parent / "fixtures" / "scanned-document-sample-3-pages.pdf"


def test_scanned_pdf_ocr():
    assert PDF_PATH.exists(), "Scanned test PDF not found"

    pdf_bytes = PDF_PATH.read_bytes()

    page_count, pages = extract_pdf_pages(pdf_bytes)

    assert page_count == 3
    assert len(pages) == 3

    for page in pages:
        assert page["extraction_method"] == "ocr"
        assert len(page["text"]) > 100


def test_scanned_pdf_key_information():
    pdf_bytes = PDF_PATH.read_bytes()

    _, pages = extract_pdf_pages(pdf_bytes)

    page_1_text = pages[0]["text"].casefold()
    page_3_text = pages[2]["text"].casefold()

    assert "99.4%" in page_1_text
    assert "4,210" in page_1_text
    assert "14 crates" in page_1_text

    assert "4401" in page_3_text
    assert "4402" in page_3_text