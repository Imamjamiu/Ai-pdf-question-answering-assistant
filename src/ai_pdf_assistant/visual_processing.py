import re
import pymupdf


def extract_visual_assets(pdf_bytes):
    """
    Extract visual-content metadata from every page of a PDF.

    Identifies:
    - Embedded images and their bounding boxes.
    - Vector drawing elements and their bounding boxes.
    - Page dimensions.

    This function does not modify the original PDF.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages_visuals = []

    try:
        for page_index, page in enumerate(document):

            images = []
            vector_drawings = []

            # -----------------------------------------
            # 1. Extract embedded image information
            # -----------------------------------------

            image_info = page.get_image_info(xrefs=True)

            for image_index, image in enumerate(image_info):

                bbox = image.get("bbox")

                if not bbox:
                    continue

                rect = pymupdf.Rect(bbox)
                # Ignore extremely small image elements.
                if rect.width < 10 or rect.height < 10:
                    continue

                images.append({
                    "asset_id": f"page_{page_index + 1}_image_{image_index + 1}",
                    "asset_type": "embedded_image",
                    "xref": image.get("xref", 0),
                    "bbox": [
                        rect.x0,
                        rect.y0,
                        rect.x1,
                        rect.y1
                    ],
                    "width": image.get("width", 0),
                    "height": image.get("height", 0)
                })

            # -----------------------------------------
            # 2. Extract vector drawing information
            # -----------------------------------------

            drawings = page.get_drawings()

            for drawing_index, drawing in enumerate(drawings):

                drawing_rect = drawing.get("rect")

                if not drawing_rect:
                    continue

                rect = pymupdf.Rect(drawing_rect)

                # Ignore very small drawing elements.
                if rect.width < 5 or rect.height < 5:
                    continue

                vector_drawings.append({
                    "asset_id": (
                        f"page_{page_index + 1}_vector_{drawing_index + 1}"
                    ),
                    "asset_type": "vector_drawing",
                    "bbox": [
                        rect.x0,
                        rect.y0,
                        rect.x1,
                        rect.y1
                    ],
                    "width": rect.width,
                    "height": rect.height
                })

            # -----------------------------------------
            # 3. Save page-level visual metadata
            # -----------------------------------------

            pages_visuals.append({
                "page_number": page_index + 1,
                "page_width": page.rect.width,
                "page_height": page.rect.height,
                "images": images,
                "vector_drawings": vector_drawings,
                "image_count": len(images),
                "vector_drawing_count": len(vector_drawings)
            })

    finally:
        document.close()

    return pages_visuals



def extract_figure_captions(pdf_bytes):
    """
    Detect figure captions and record their page positions.

    Supports captions such as:
    Figure 2.2
    Figure 10.1: Page Replacement
    Fig. 3.4 - Neural Network Architecture
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    figure_captions = []

    caption_pattern = re.compile(
        r"^\s*(?:Figure|Fig\.?)\s+\d+(?:\.\d+)*"
        r"(?:\s*[:\-–—]\s*.+)?\s*$",
        re.IGNORECASE
    )

    try:
        for page_index, page in enumerate(document):

            text_blocks = page.get_text("dict")["blocks"]

            for block in text_blocks:

                if "lines" not in block:
                    continue

                for line in block["lines"]:

                    spans = line.get("spans", [])

                    line_text = "".join(
                        span.get("text", "")
                        for span in spans
                    ).strip()

                    if not line_text:
                        continue

                    if not caption_pattern.match(line_text):
                        continue

                    bbox = line.get("bbox")

                    if not bbox:
                        continue

                    figure_captions.append({
                        "caption_id": (
                            f"page_{page_index + 1}_caption_"
                            f"{len(figure_captions) + 1}"
                        ),
                        "page_number": page_index + 1,
                        "caption": line_text,
                        "bbox": list(bbox)
                    })

    finally:
        document.close()

    return figure_captions




def detect_figure_regions(pdf_bytes, captions=None, padding=12):
    """
    Estimate complete figure regions using caption positions,
    surrounding text blocks, embedded images, and vector drawings.

    Returns figure metadata with page number and bounding box.
    """

    if captions is None:
        captions = extract_figure_captions(pdf_bytes)

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    detected_regions = []

    try:
        for caption in captions:

            page_number = caption["page_number"]
            page = document[page_number - 1]

            caption_rect = pymupdf.Rect(caption["bbox"])

            # -----------------------------------------
            # 1. Collect visual element boundaries
            # -----------------------------------------

            visual_rects = []

            for drawing in page.get_drawings():

                drawing_bbox = drawing.get("rect")

                if not drawing_bbox:
                    continue

                rect = pymupdf.Rect(drawing_bbox)

                if rect.width < 2 or rect.height < 2:
                    continue

                visual_rects.append(rect)

            for image in page.get_image_info(xrefs=True):

                image_bbox = image.get("bbox")

                if not image_bbox:
                    continue

                rect = pymupdf.Rect(image_bbox)

                if rect.width < 2 or rect.height < 2:
                    continue

                visual_rects.append(rect)

            # Use the closest preceding figure caption as the lower boundary.
            # Diagram labels should not determine this boundary because
            # they may be extracted as ordinary text blocks.

            previous_captions = [
                item for item in captions
                if item["page_number"] == page_number
                   and item["bbox"][3] < caption_rect.y0
            ]

            if previous_captions:
                previous_caption = max(
                    previous_captions,
                    key=lambda item: item["bbox"][3]
                )
                start_y = previous_caption["bbox"][3] + 4
            else:
                start_y = 0

            # -----------------------------------------
            # 3. Collect visual elements in the region
            # -----------------------------------------

            figure_elements = []

            for rect in visual_rects:

                if rect.y1 <= start_y:
                    continue

                if rect.y0 >= caption_rect.y0:
                    continue

                figure_elements.append(rect)

            # -----------------------------------------
            # 4. Calculate the combined bounding box
            # -----------------------------------------

            if not figure_elements:

                detected_regions.append({
                    **caption,
                    "bbox": None,
                    "status": "no_visual_elements_found"
                })

                continue

            combined_rect = pymupdf.Rect(figure_elements[0])

            for rect in figure_elements[1:]:
                combined_rect.include_rect(rect)

            # Include the complete caption text block.
            for block in page.get_text("blocks"):

                block_rect = pymupdf.Rect(block[:4])

                if block_rect.intersects(caption_rect):
                    combined_rect.include_rect(block_rect)
                    break

            # Add padding and keep the crop inside the page.
            combined_rect = pymupdf.Rect(
                combined_rect.x0 - padding,
                combined_rect.y0 - padding,
                combined_rect.x1 + padding,
                combined_rect.y1 + padding
            )

            combined_rect = combined_rect & page.rect

            detected_regions.append({
                "caption_id": caption["caption_id"],
                "caption": caption["caption"],
                "page_number": page_number,
                "bbox": [
                    combined_rect.x0,
                    combined_rect.y0,
                    combined_rect.x1,
                    combined_rect.y1
                ],
                "visual_element_count": len(figure_elements),
                "status": "detected"
            })

    finally:
        document.close()

    return detected_regions





def render_page_region(pdf_bytes, page_number, bbox=None, scale=2):
    """
    Render an entire PDF page or a selected rectangular region.

    Args:
        pdf_bytes: Original PDF file as bytes.
        page_number: One-based PDF page number.
        bbox: Optional [x0, y0, x1, y1] coordinates.
        scale: Rendering resolution multiplier.

    Returns:
        PNG image as bytes.
    """

    if page_number < 1:
        raise ValueError("Page numbers start at 1.")

    if scale <= 0:
        raise ValueError("Scale must be greater than zero.")

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:
        if page_number > len(document):
            raise ValueError(
                f"Page {page_number} does not exist. "
                f"This PDF has {len(document)} pages."
            )

        page = document[page_number - 1]

        clip = None

        if bbox is not None:
            clip = pymupdf.Rect(bbox) & page.rect

            if clip.is_empty or clip.is_infinite:
                raise ValueError("The selected region is outside the page.")

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            clip=clip,
            alpha=False
        )

        return pixmap.tobytes("png")

    finally:
        document.close()
