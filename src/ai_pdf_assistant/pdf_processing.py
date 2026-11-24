import io, os
import pymupdf, pdfplumber, pytesseract
from PIL import Image

# Tesseract configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# Clean individual PDF pages


def clean_pdf_page(page, page_number):
    """Keep the original PDF page intact."""
    return page

# OCR helper functions
def run_ocr(image):
    """Preprocess an image using Pillow before Tesseract OCR."""
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps
        image = ImageOps.grayscale(image)
        image = ImageEnhance.Contrast(image).enhance(1.8)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.filter(ImageFilter.SHARPEN)
        return pytesseract.image_to_string(image, config="--oem 3 --psm 6").strip()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError("Tesseract OCR was not found. Install Tesseract and configure its executable path.") from error

def render_page_for_ocr(page):
    """Render a PDF page into an image for OCR."""
    scale = 3
    matrix = pymupdf.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png")))

def extract_embedded_image_text(pdf_document, page):
    """Extract OCR text from meaningful embedded images on a PDF page."""
    extracted_text, processed_xrefs = [], set()
    for image_info in page.get_images(full=True):
        xref, width, height = image_info[0], image_info[2], image_info[3]
        if width < 100 or height < 60 or width * height < 12000 or xref in processed_xrefs:
            continue
        processed_xrefs.add(xref)
        try:
            image_data = pdf_document.extract_image(xref)
            image = Image.open(io.BytesIO(image_data["image"])).convert("RGB")
            image_text = run_ocr(image)
            if image_text:
                extracted_text.append(image_text)
        except pytesseract.TesseractNotFoundError:
            raise
        except Exception:
            continue
    return "\n".join(extracted_text)


# Avoid duplicate OCR/native text
def merge_extracted_text(native_text, ocr_text):
    """Combine native PDF text and OCR text while avoiding obvious duplicate lines."""
    native_text, ocr_text = native_text.strip(), ocr_text.strip()
    if not native_text:
        return ocr_text
    if not ocr_text:
        return native_text
    existing_lines = {line.strip().casefold() for line in native_text.splitlines() if line.strip()}
    additional_lines = [line.strip() for line in ocr_text.splitlines() if line.strip() and line.strip().casefold() not in existing_lines]
    return native_text + "\n" + "\n".join(additional_lines) if additional_lines else native_text


# Extract text from all PDF pages
def extract_pdf_pages(pdf_bytes):
    """Extract text from every PDF page."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            for page_number, page in enumerate(pdf.pages, start=1):
                cleaned_page = clean_pdf_page(page, page_number)
                native_text = (cleaned_page.extract_text(x_tolerance=1.5, y_tolerance=3) or "").strip()
                ocr_text = ""
                image_page = pdf_document[page_number - 1]
                if sum(char.isalnum() for char in native_text) < 30:
                    page_image = render_page_for_ocr(image_page)
                    ocr_text = run_ocr(page_image)
                else:
                    embedded_image_text = extract_embedded_image_text(pdf_document, image_page)
                    if embedded_image_text:
                        ocr_text = merge_extracted_text(ocr_text, embedded_image_text)
                final_text = merge_extracted_text(native_text, ocr_text)
                extraction_method = "mixed" if native_text and ocr_text else "native" if native_text else "ocr" if ocr_text else "none"
                pages.append({"page_number": page_number, "text": final_text, "extraction_method": extraction_method})
        finally:
            pdf_document.close()
    return page_count, pages


# Maintain the existing text extraction function
def extract_pdf_text(pdf_bytes):
    """Maintain backward compatibility with existing code."""
    page_count, pages = extract_pdf_pages(pdf_bytes)
    full_text = "\n".join(page["text"] for page in pages if page["text"].strip())
    return page_count, full_text