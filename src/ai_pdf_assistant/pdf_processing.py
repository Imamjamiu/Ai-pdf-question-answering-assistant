import io
from difflib import SequenceMatcher

import pdfplumber


class OcrDependencyError(RuntimeError):
    """Raised when OCR is needed but its Python or system dependency is missing."""


def clean_pdf_page(page, page_number):
    # Keep the page intact. Fixed coordinates can remove real text when PDFs
    # use different page sizes or layouts; page_number is retained separately.
    return page


def _normalise_line(line):
    return " ".join(line.casefold().split()).strip(" .,:;|")


def _merge_native_and_ocr_text(native_text, ocr_text):
    """Keep native text first and append OCR lines not already represented."""
    native_text = (native_text or "").strip()
    ocr_lines = [line.strip() for line in (ocr_text or "").splitlines() if line.strip()]

    if not native_text:
        return "\n".join(ocr_lines)

    native_lines = [line.strip() for line in native_text.splitlines() if line.strip()]
    known_lines = [_normalise_line(line) for line in native_lines]
    additions = []

    for line in ocr_lines:
        normalised = _normalise_line(line)
        if not normalised:
            continue

        duplicate = any(
            normalised == existing
            or (
                min(len(normalised), len(existing)) >= 12
                and SequenceMatcher(None, normalised, existing).ratio() >= 0.90
            )
            for existing in known_lines
        )
        if not duplicate:
            additions.append(line)
            known_lines.append(normalised)

    if not additions:
        return native_text

    return native_text + "\n" + "\n".join(additions)


def _get_image_regions(page, minimum_area_ratio=0.01):
    """Return meaningful embedded-image bounds in pdfplumber's top-left coordinates."""
    page_area = max(float(page.width) * float(page.height), 1.0)
    regions = []

    for image in getattr(page, "images", []) or []:
        try:
            x0 = float(image["x0"])
            top = float(image["top"])
            x1 = float(image["x1"])
            bottom = float(image["bottom"])
        except (KeyError, TypeError, ValueError):
            continue

        width = max(0.0, x1 - x0)
        height = max(0.0, bottom - top)
        if width * height / page_area < minimum_area_ratio:
            continue

        regions.append((x0, top, x1, bottom))

    return regions


def _ocr_region(fitz_page, clip, dpi, language, page_number):
    """Render a full page or image crop and pass it to the local Tesseract engine."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise OcrDependencyError(
            "OCR is required for this PDF page, but an OCR dependency is missing. "
            "Install the project requirements and the Tesseract OCR application."
        ) from error

    try:
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pixmap = fitz_page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        page_segmentation = 3 if clip is None else 11
        return pytesseract.image_to_string(
            image,
            lang=language,
            config=f"--psm {page_segmentation}",
        ).strip()
    except Exception as error:
        # Keep a useful installation message instead of exposing a long
        # low-level Tesseract traceback to the Streamlit user.
        tesseract_error = getattr(
            getattr(pytesseract, "pytesseract", None),
            "TesseractNotFoundError",
            (),
        )
        if tesseract_error and isinstance(error, tesseract_error):
            raise OcrDependencyError(
                "Tesseract OCR is not installed or is not on PATH. "
                "Install Tesseract and restart the application."
            ) from error
        raise RuntimeError(
            f"OCR failed on PDF page {page_number}: {error}"
        ) from error


def extract_pdf_pages(
    pdf_bytes,
    *,
    enable_ocr=True,
    min_native_chars=40,
    ocr_dpi=250,
    ocr_language="eng",
):
    """Extract page text, using Tesseract for scanned and image-containing pages.

    Native PDF text remains the primary source. Pages with very little native
    text are OCRed in full. On pages with sufficient native text, meaningful
    embedded images are OCRed separately so text inside figures/screenshots is
    not silently ignored. Each returned page keeps its original PDF page number.

    The existing return contract is preserved: (page_count, pages). Each page
    retains the required page_number/text keys and adds extraction metadata.
    """
    pages = []
    fitz_document = None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)

            for page_number, page in enumerate(pdf.pages, start=1):
                cleaned_page = clean_pdf_page(page, page_number)
                native_text = (
                    cleaned_page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
                ).strip()
                image_regions = _get_image_regions(cleaned_page)
                native_is_sparse = len(native_text) < min_native_chars
                should_ocr_full_page = enable_ocr and native_is_sparse
                should_ocr_images = enable_ocr and not native_is_sparse and bool(image_regions)

                ocr_text_parts = []

                if should_ocr_full_page or should_ocr_images:
                    try:
                        import fitz
                    except ImportError as error:
                        raise OcrDependencyError(
                            "PyMuPDF is required to render PDF pages for OCR. "
                            "Install the project requirements."
                        ) from error

                    if fitz_document is None:
                        fitz_document = fitz.open(
                            stream=pdf_bytes,
                            filetype="pdf",
                        )

                    fitz_page = fitz_document[page_number - 1]

                    if should_ocr_full_page:
                        ocr_text = _ocr_region(
                            fitz_page,
                            clip=None,
                            dpi=ocr_dpi,
                            language=ocr_language,
                            page_number=page_number,
                        )
                        if ocr_text:
                            ocr_text_parts.append(ocr_text)
                    else:
                        for x0, top, x1, bottom in image_regions:
                            # pdfplumber and PyMuPDF both use top-left page
                            # coordinates for these page-space rectangles.
                            clip = fitz.Rect(x0, top, x1, bottom)
                            ocr_text = _ocr_region(
                                fitz_page,
                                clip=clip,
                                dpi=ocr_dpi,
                                language=ocr_language,
                                page_number=page_number,
                            )
                            if ocr_text:
                                ocr_text_parts.append(ocr_text)

                ocr_text = "\n".join(ocr_text_parts).strip()
                combined_text = _merge_native_and_ocr_text(native_text, ocr_text)

                if not combined_text:
                    continue

                if native_text and ocr_text:
                    extraction_method = "mixed"
                elif ocr_text:
                    extraction_method = "ocr"
                else:
                    extraction_method = "native"

                pages.append({
                    "page_number": page_number,
                    "text": combined_text,
                    "extraction_method": extraction_method,
                    "native_text_chars": len(native_text),
                    "ocr_text_chars": len(ocr_text),
                })

        return page_count, pages
    finally:
        if fitz_document is not None:
            fitz_document.close()


def extract_pdf_text(pdf_bytes):
    page_count, pages = extract_pdf_pages(pdf_bytes)
    full_text = "\n".join(page["text"] for page in pages)
    return page_count, full_text
