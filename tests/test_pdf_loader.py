import pytest
from src.pdf_loader import extract_pages


def test_invalid_pdf_is_rejected():
    with pytest.raises(ValueError, match="readable PDF"):
        extract_pages("not-a-pdf.pdf", b"plain text")
