"""PDF text extraction (PyMuPDF) with automatic OCR fallback (Tesseract) and chunking."""
import io
import re
from typing import Dict, List

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:  # older versions
    import fitz

from . import config

_ocr_ready = None


def _ocr_available() -> bool:
    global _ocr_ready
    if _ocr_ready is not None:
        return _ocr_ready
    try:
        import pytesseract

        if config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        pytesseract.get_tesseract_version()
        _ocr_ready = True
    except Exception:
        _ocr_ready = False
    return _ocr_ready


def _ocr_page(page) -> str:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate line breaks
    text = text.replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path, use_ocr: bool = True) -> List[Dict]:
    """Return [{'page': int, 'text': str, 'ocr': bool}, ...] (1-based pages)."""
    pages, ocr_pages, skipped_ocr = [], 0, 0
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            text, used_ocr = page.get_text("text") or "", False
            if len(text.strip()) < config.OCR_MIN_CHARS and use_ocr:
                if _ocr_available():
                    try:
                        text, used_ocr = _ocr_page(page), True
                        ocr_pages += 1
                    except Exception:
                        pass
                else:
                    skipped_ocr += 1
            text = clean_text(text)
            if text:
                pages.append({"page": i, "text": text, "ocr": used_ocr})
    if ocr_pages:
        print(f"    OCR used on {ocr_pages} page(s)")
    if skipped_ocr:
        print(f"    {skipped_ocr} page(s) had no text and Tesseract is not installed "
              f"(see README for OCR setup)")
    return pages


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Split on sentence and paragraph boundaries, retaining a small overlap."""
    sentences = re.split(r"(?<=[.!?।])\s+|\n+", text)
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        while len(s) > size:  # very long run-on text
            if current:
                chunks.append(current)
                current = ""
            chunks.append(s[:size])
            s = s[size - overlap:]
        if len(current) + len(s) + 1 <= size:
            current = f"{current} {s}".strip()
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {s}".strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 80]


_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S")
_HEADING_JOINERS = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to"}


def _is_heading(line: str) -> bool:
    if not line or len(line) > 120 or re.search(r"[.!?।]$", line):
        return False
    words = re.findall(r"[A-Za-z]+", line)
    uppercase = sum(word.isupper() for word in words)
    title_words = [word for word in words if word.lower() not in _HEADING_JOINERS]
    title_case = sum(word.istitle() for word in title_words)
    return bool(_NUMBERED_HEADING.match(line)) or (
        len(words) <= 12 and words and title_words
        and (uppercase / len(words) >= 0.7 or title_case / len(title_words) >= 0.8)
    )


def chunk_page(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[Dict]:
    """Return heading-aware chunks carrying the heading as section metadata."""
    sections = []
    heading, body = "General", []

    def flush():
        content = "\n".join(body).strip()
        if not content:
            return
        for chunk in chunk_text(content, size=size, overlap=overlap):
            sections.append({"text": f"{heading}\n{chunk}" if heading != "General" else chunk,
                             "section": heading})

    for line in text.splitlines():
        line = line.strip()
        if not line:
            if body and body[-1] != "":
                body.append("")
            continue
        if _is_heading(line):
            flush()
            body.clear()
            heading = line
        else:
            body.append(line)
    flush()
    return sections
