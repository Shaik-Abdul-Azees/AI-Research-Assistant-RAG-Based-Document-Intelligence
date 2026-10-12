from llm import generate_answer
from ingest import extract_pdf_text, chunk_pages
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small, efficient embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_embeddings(chunks):
    """Convert text chunks into numerical vectors."""
    if not chunks:
        raise ValueError("No chunks provided.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Created embeddings for {len(chunks)} chunks.")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    return embeddings.astype(np.float32)


def build_faiss_index(embeddings):
    """Build a FAISS index for normalized text embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    embeddings = np.ascontiguousarray(
        embeddings, dtype=np.float32
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")
    return index


def search_chunks(query, chunks, index, top_k=3):
    """Retrieve the most similar text chunks for a query."""
    if not chunks:
        return []

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(
        np.ascontiguousarray(query_vector),
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "score": float(score)
        })

    return results





from ingest import extract_pdf_text, chunk_pages
from pathlib import Path



if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ").strip().strip('"')

    # Extract and chunk the PDF
    pages = extract_pdf_text(pdf_path)
    chunks = chunk_pages(pages, chunk_size=500, overlap=100)

    print(f"Extracted {len(pages)} pages.")
    print(f"Created {len(chunks)} chunks.")

    # Build the vector index
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    # Ask questions using RAG
    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        results = search_chunks(question, chunks, index, top_k=3)

        print("\nGenerating answer...")
        answer = generate_answer(question, results)

        print("\nAI Answer:")
        print(answer)
