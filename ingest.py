from pathlib import Path
import pymupdf



def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    pages = []
    with pymupdf.open(path) as document: 
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "This may be a scanned PDF requiring OCR."
        )

    return pages


def extract_pdf_text_from_bytes(pdf_bytes):
    """Extract text and page numbers from an uploaded PDF."""
    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No extractable text found. Scanned PDFs may require OCR."
        )

    return pages



def chunk_pages(pages, chunk_size=500, overlap=100):
    """Split page text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    chunks = []

    for item in pages:
        text = item["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append({
                "page": item["page"],
                "text": text[start:end]
            })

            if end == len(text):
                break

            start = end - overlap

    return chunks



if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF: ").strip().strip('"')

    try:
        pages = extract_pdf_text(pdf_path)

        chunks = chunk_pages(pages)

        print(f"Created {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {i} | Page {chunk['page']}")
            print(chunk["text"])
            print(f"\nSuccessfully extracted {len(pages)} pages.")

        for item in pages[:2]:
            print(f"\n--- Page {item['page']} ---")
            print(item["text"][:500])

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
