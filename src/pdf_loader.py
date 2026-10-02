from dataclasses import dataclass
import pymupdf as fitz


@dataclass(frozen=True)
class PageText:
    document: str
    page: int
    text: str


def extract_pages(filename: str, raw: bytes) -> list[PageText]:
    """Extract non-empty page text and retain the original page number."""
    try:
        pdf = fitz.open(stream=raw, filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as exc:
        raise ValueError(f"{filename} is not a readable PDF.") from exc
    pages = [PageText(filename, index + 1, page.get_text("text").strip())
             for index, page in enumerate(pdf) if page.get_text("text").strip()]
    pdf.close()
    if not pages:
        raise ValueError(f"{filename} contains no extractable text.")
    return pages
