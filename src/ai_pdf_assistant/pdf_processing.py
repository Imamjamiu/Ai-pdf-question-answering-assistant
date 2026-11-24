
import io
import os
import pymupdf
import pdfplumber
import pytesseract
from PIL import Image

# --------------------------------------------------
# 1. Tesseract configuration
# --------------------------------------------------

# Windows installation path.
# If Tesseract is available through PATH, that will also work.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# --------------------------------------------------
# 2. Clean individual PDF pages
# --------------------------------------------------

def clean_pdf_page(page, page_number):
    """
    Keep the original PDF page intact.

    Page numbers are maintained separately as metadata.
    """
    return page


# --------------------------------------------------
# 3. OCR helper functions
# --------------------------------------------------

def run_ocr(image):
    """
    Preprocess an image using Pillow before Tesseract OCR.
    """

    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps

        # Convert to grayscale.
        image = ImageOps.grayscale(image)

        # Improve contrast.
        image = ImageEnhance.Contrast(image).enhance(1.8)

        # Reduce small image noise.
        image = image.filter(
            ImageFilter.MedianFilter(size=3)
        )

        # Improve text edge definition.
        image = image.filter(
            ImageFilter.SHARPEN
        )

        return pytesseract.image_to_string(
            image,
            config="--oem 3 --psm 6"
        ).strip()

    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR was not found. "
            "Install Tesseract and configure its executable path."
        ) from error


def render_page_for_ocr(page):
    """
    Render a PDF page into an image for OCR.
    """

    scale = 3

    matrix = pymupdf.Matrix(scale, scale)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = Image.open(
        io.BytesIO(pixmap.tobytes("png"))
    )

    return image


def extract_embedded_image_text(pdf_document, page):
    """
    Extract OCR text from meaningful embedded images on a PDF page.

    This helps recover text inside figures, screenshots and
    image-based tables, even when the page has selectable text.
    """

    extracted_text = []

    processed_xrefs = set()

    for image_info in page.get_images(full=True):

        xref = image_info[0]
        width = image_info[2]
        height = image_info[3]

        # Avoid processing tiny icons, logos and decorative images.
        if width < 100 or height < 60:
            continue

        if width * height < 12000:
            continue

        # Avoid processing the same embedded image more than once.
        if xref in processed_xrefs:
            continue

        processed_xrefs.add(xref)

        try:
            image_data = pdf_document.extract_image(xref)

            image = Image.open(
                io.BytesIO(image_data["image"])
            ).convert("RGB")

            image_text = run_ocr(image)

            if image_text:
                extracted_text.append(image_text)

        except pytesseract.TesseractNotFoundError:
            raise

        except Exception:
            # A damaged or unsupported embedded image should not
            # prevent extraction of the rest of the PDF.
            continue

    return "\n".join(extracted_text)


# --------------------------------------------------
# 4. Avoid duplicate OCR/native text
# --------------------------------------------------

def merge_extracted_text(native_text, ocr_text):
    """
    Combine native PDF text and OCR text while avoiding
    obvious duplicate lines.
    """

    native_text = native_text.strip()
    ocr_text = ocr_text.strip()

    if not native_text:
        return ocr_text

    if not ocr_text:
        return native_text

    existing_lines = {
        line.strip().casefold()
        for line in native_text.splitlines()
        if line.strip()
    }

    additional_lines = []

    for line in ocr_text.splitlines():

        cleaned_line = line.strip()

        if not cleaned_line:
            continue

        if cleaned_line.casefold() not in existing_lines:
            additional_lines.append(cleaned_line)

            existing_lines.add(cleaned_line.casefold())

    if additional_lines:
        return native_text + "\n" + "\n".join(additional_lines)

    return native_text


# --------------------------------------------------
# 5. Extract text from all PDF pages
# --------------------------------------------------

def extract_pdf_pages(pdf_bytes):
    """
    Extract text from every PDF page.

    Extraction strategy:

    1. Extract selectable text using pdfplumber.
    2. If native text is insufficient, OCR the entire page.
    3. OCR meaningful embedded images to recover text
       from figures and image-based content.
    4. Preserve original PDF page numbers.
    """

    pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        page_count = len(pdf.pages)

        # Open the same PDF with PyMuPDF for image extraction
        # and page rendering.

        pdf_document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        try:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                cleaned_page = clean_pdf_page(
                    page,
                    page_number
                )

                native_text = cleaned_page.extract_text(
                    x_tolerance=1.5,
                    y_tolerance=3
                ) or ""

                native_text = native_text.strip()

                ocr_text = ""

                # Corresponding PyMuPDF page uses zero-based indexing.
                image_page = pdf_document[page_number - 1]

                # Check whether native extraction produced enough text.
                alphanumeric_count = sum(
                    character.isalnum()
                    for character in native_text
                )

                if alphanumeric_count < 30:

                    # Scanned page or page with insufficient selectable text.
                    # OCR the entire page.
                    page_image = render_page_for_ocr(image_page)
                    ocr_text = run_ocr(page_image)

                else:

                    # The page already contains sufficient native text.
                    # Inspect embedded images for additional text in figures,
                    # screenshots and image-based tables.
                    embedded_image_text = extract_embedded_image_text(
                        pdf_document,
                        image_page
                    )

                    if embedded_image_text:
                        ocr_text = merge_extracted_text(
                            ocr_text,
                            embedded_image_text
                        )

                final_text = merge_extracted_text(
                    native_text,
                    ocr_text
                )

                # Identify how the page text was obtained.
                if native_text and ocr_text:
                    extraction_method = "mixed"

                elif native_text:
                    extraction_method = "native"

                elif ocr_text:
                    extraction_method = "ocr"

                else:
                    extraction_method = "none"

                # Retain every page, including pages without readable text.
                pages.append({
                    "page_number": page_number,
                    "text": final_text,
                    "extraction_method": extraction_method
                })

        finally:
            pdf_document.close()

    return page_count, pages


# --------------------------------------------------
# 6. Maintain the existing text extraction function
# --------------------------------------------------

def extract_pdf_text(pdf_bytes):
    """
    Maintain backward compatibility with existing code.
    """

    page_count, pages = extract_pdf_pages(pdf_bytes)

    full_text = "\n".join(
        page["text"]
        for page in pages
        if page["text"].strip()
    )

    return page_count, full_text
