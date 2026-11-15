
import io
import pdfplumber


# --------------------------------------------------
# 1. Clean individual PDF pages
# --------------------------------------------------

def clean_pdf_page(page, page_number):
    # Keep the page intact. Fixed coordinates can remove real text when PDFs
    # use different page sizes or layouts; page_number is retained separately
    # as metadata by extract_pdf_pages().
    return page


# --------------------------------------------------
# 2. Extract text from the PDF
# --------------------------------------------------

# --------------------------------------------------
# 2. Extract text while preserving page numbers
# --------------------------------------------------

def extract_pdf_pages(pdf_bytes):

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        page_count = len(pdf.pages)

        pages = []

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            cleaned_page = clean_pdf_page(
                page,
                page_number
            )

            page_text = cleaned_page.extract_text(
                x_tolerance=1.5,
                y_tolerance=3
            ) or ""

            if page_text.strip():

                pages.append({
                    "page_number": page_number,
                    "text": page_text.strip()
                })

    return page_count, pages


# --------------------------------------------------
# 3. Maintain the existing text extraction function
# --------------------------------------------------

def extract_pdf_text(pdf_bytes):

    page_count, pages = extract_pdf_pages(pdf_bytes)

    full_text = "\n".join(
        page["text"] for page in pages
    )

    return page_count, full_text
