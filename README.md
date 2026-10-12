
# AI Research Assistant — RAG-Based Document Intelligence

A Python-based application that lets users upload PDF documents, ask questions in natural language, and receive AI-generated answers grounded in retrieved document passages.

## Features

- PDF text extraction with page numbers
- Text chunking for document indexing
- Semantic embeddings using Sentence Transformers
- Similarity search using FAISS
- Local answer generation using Ollama and Qwen2.5
- Interactive Streamlit interface
- Source passages, page citations, and similarity scores
- Document reset and multiple-question support

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Ollama
- Qwen2.5

## Architecture

1. Upload a PDF through the Streamlit interface.
2. Extract text and page numbers using PyMuPDF.
3. Split the text into searchable chunks.
4. Generate embeddings using Sentence Transformers.
5. Store the vectors in a FAISS index.
6. Convert a user's question into an embedding.
7. Retrieve relevant document chunks.
8. Send the question and retrieved context to Qwen2.5 through Ollama.
9. Display the answer and supporting source passages.

## Prerequisites

- Python 3.11
- Ollama installed
- Git

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the local language model:

```powershell
ollama pull qwen2.5:3b
```

Ensure Ollama is running, then start the application:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Usage

1. Upload a text-based PDF.
2. Click **Process PDF**.
3. Enter a question about the document.
4. Review the answer and supporting passages.
5. Ask another question or reset the document.

## Testing

Run the automated tests with:

```powershell
python -m pytest -v
```

## Limitations

- Scanned PDFs may require OCR.
- Answer quality depends on document extraction, retrieval quality, and model behavior.
- Page citations identify retrieved passages; they do not independently guarantee that every generated claim is supported.
- The local language model requires sufficient system resources.

## Future Improvements

- Sentence-aware chunking
- Retrieval relevance thresholds
- Improved citation validation
- Automated retrieval evaluation
- Document history and multi-document search
- API support with FastAPI

## License

Choose and add an appropriate open-source license before distributing the project.

