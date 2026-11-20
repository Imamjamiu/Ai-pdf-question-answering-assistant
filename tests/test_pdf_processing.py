import sys
import types

import pytest

from ai_pdf_assistant import pdf_processing


class FakePixmap:
    def tobytes(self, _format):
        return b"fake-png"


class FakeFitzPage:
    def __init__(self):
        self.clips = []

    def get_pixmap(self, matrix, clip, alpha):
        self.clips.append(clip)
        return FakePixmap()


class FakeFitzDocument:
    def __init__(self, page_count):
        self.pages = [FakeFitzPage() for _ in range(page_count)]
        self.closed = False

    def close(self):
        self.closed = True


class FakePdfPage:
    width = 600
    height = 800

    def __init__(self, text="", images=None):
        self.text = text
        self.images = images or []

    def extract_text(self, **_kwargs):
        return self.text


class FakePdfDocument:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def install_ocr_stubs(monkeypatch, page_count, recognised_text):
    fitz_document = FakeFitzDocument(page_count)

    class FakeRect:
        def __init__(self, *coords):
            self.coords = coords

    fake_fitz = types.SimpleNamespace(
        Matrix=lambda *_args: object(),
        Rect=FakeRect,
        open=lambda **_kwargs: fitz_document,
    )
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = types.SimpleNamespace(open=lambda _stream: object())
    fake_pytesseract = types.SimpleNamespace(
        image_to_string=lambda *_args, **_kwargs: recognised_text
    )

    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    return fitz_document


def test_merge_native_and_ocr_text_avoids_exact_duplicate_lines():
    merged = pdf_processing._merge_native_and_ocr_text(
        "Virtual memory maps pages.\nPage tables store mappings.",
        "Virtual memory maps pages.\nA page fault occurs when a page is absent.",
    )
    assert merged.count("Virtual memory maps pages.") == 1
    assert "A page fault occurs when a page is absent." in merged


def test_get_image_regions_ignores_tiny_images():
    page = types.SimpleNamespace(
        width=100,
        height=100,
        images=[
            {"x0": 0, "top": 0, "x1": 5, "bottom": 5},
            {"x0": 10, "top": 10, "x1": 80, "bottom": 80},
        ],
    )
    assert pdf_processing._get_image_regions(page) == [(10.0, 10.0, 80.0, 80.0)]


def test_text_pdf_uses_native_extraction_without_loading_ocr(monkeypatch):
    fake_pdf = FakePdfDocument([
        FakePdfPage("This is a sufficiently long native PDF text passage.")
    ])
    monkeypatch.setattr(pdf_processing.pdfplumber, "open", lambda *_args: fake_pdf)
    page_count, pages = pdf_processing.extract_pdf_pages(b"pdf-bytes")
    assert page_count == 1
    assert pages[0]["page_number"] == 1
    assert pages[0]["extraction_method"] == "native"
    assert pages[0]["text"].startswith("This is a sufficiently long")


def test_scanned_page_uses_full_page_ocr_and_preserves_page_number(monkeypatch):
    fake_pdf = FakePdfDocument([FakePdfPage(""), FakePdfPage("")])
    monkeypatch.setattr(pdf_processing.pdfplumber, "open", lambda *_args: fake_pdf)
    fitz_document = install_ocr_stubs(
        monkeypatch, page_count=2,
        recognised_text="Virtual memory allows processes to use virtual addresses.",
    )
    page_count, pages = pdf_processing.extract_pdf_pages(b"pdf-bytes")
    assert page_count == 2
    assert len(pages) == 2
    assert [page["page_number"] for page in pages] == [1, 2]
    assert all(page["extraction_method"] == "ocr" for page in pages)
    assert fitz_document.closed


def test_mixed_page_ocr_scans_embedded_image_region(monkeypatch):
    fake_pdf = FakePdfDocument([
        FakePdfPage(
            "This page has enough selectable text to avoid full page OCR.",
            images=[{"x0": 20, "top": 100, "x1": 400, "bottom": 500}],
        )
    ])
    monkeypatch.setattr(pdf_processing.pdfplumber, "open", lambda *_args: fake_pdf)
    fitz_document = install_ocr_stubs(
        monkeypatch, page_count=1,
        recognised_text="Diagram label: page table entry.",
    )
    page_count, pages = pdf_processing.extract_pdf_pages(b"pdf-bytes")
    assert page_count == 1
    assert pages[0]["extraction_method"] == "mixed"
    assert "selectable text" in pages[0]["text"]
    assert "Diagram label: page table entry." in pages[0]["text"]
    assert fitz_document.pages[0].clips[0] is not None


def test_ocr_can_be_disabled_for_native_only_extraction(monkeypatch):
    fake_pdf = FakePdfDocument([FakePdfPage("Short title.")])
    monkeypatch.setattr(pdf_processing.pdfplumber, "open", lambda *_args: fake_pdf)
    page_count, pages = pdf_processing.extract_pdf_pages(b"pdf-bytes", enable_ocr=False)
    assert page_count == 1
    assert pages[0]["text"] == "Short title."
    assert pages[0]["extraction_method"] == "native"


def test_missing_ocr_dependency_has_actionable_error(monkeypatch):
    fake_pdf = FakePdfDocument([FakePdfPage("")])
    monkeypatch.setattr(pdf_processing.pdfplumber, "open", lambda *_args: fake_pdf)
    monkeypatch.delitem(sys.modules, "fitz", raising=False)
    real_import = __import__

    def import_without_fitz(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_fitz)
    with pytest.raises(pdf_processing.OcrDependencyError, match="PyMuPDF"):
        pdf_processing.extract_pdf_pages(b"pdf-bytes")
