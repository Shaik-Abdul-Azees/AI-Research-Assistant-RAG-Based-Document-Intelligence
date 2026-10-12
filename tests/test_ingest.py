
import pymupdf

from ingest import extract_pdf_text_from_bytes, chunk_pages


def create_test_pdf():
    """Create a small PDF in memory for testing."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "The lighthouse keeper remembered autumn. "
        "The astronomer studied distant stars.",
    )
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_pdf_text_from_bytes():
    pdf_bytes = create_test_pdf()

    pages = extract_pdf_text_from_bytes(pdf_bytes)

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "lighthouse keeper" in pages[0]["text"]


def test_chunk_pages_preserves_page_number():
    pages = [
        {
            "page": 2,
            "text": (
                "The lighthouse keeper remembered autumn. "
                "The astronomer studied distant stars."
            ),
        }
    ]

    chunks = chunk_pages(pages, chunk_size=500, overlap=100)

    assert len(chunks) >= 1
    assert all(chunk["page"] == 2 for chunk in chunks)
    assert any("lighthouse keeper" in chunk["text"] for chunk in chunks)
