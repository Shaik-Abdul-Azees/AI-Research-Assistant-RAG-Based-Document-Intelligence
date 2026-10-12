
import streamlit as st

from ingest import extract_pdf_text_from_bytes, chunk_pages
from retrieval import (
    create_embeddings,
    build_faiss_index,
    search_chunks,
)
from llm import generate_answer


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.4rem;
    }

    .muted {
        opacity: 0.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📚 Research Assistant")
    st.caption("Document intelligence powered by local AI")
    st.divider()

    st.subheader("About")
    st.write(
        "Upload a PDF, ask questions in natural language, "
        "and inspect the passages used to generate answers."
    )

    st.divider()
    st.subheader("Technology")
    st.markdown(
        "- Python\n"
        "- Sentence Transformers\n"
        "- FAISS vector search\n"
        "- Ollama + Qwen2.5\n"
        "- Streamlit"
    )

    if "index" in st.session_state:
        st.divider()
        st.success("Document indexed")

        st.caption(
            f"File: {st.session_state['filename']}"
        )
        st.caption(
            f"Pages: {len(st.session_state['pages'])}"
        )
        st.caption(
            f"Chunks: {len(st.session_state['chunks'])}"
        )

        if st.button("🗑️ Reset document", use_container_width=True):
            for key in [
                "pages",
                "chunks",
                "index",
                "filename",
                "last_question",
                "last_answer",
                "last_results",
            ]:
                st.session_state.pop(key, None)

            st.rerun()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>AI Research Assistant</h1>
        <p class="muted">
            Explore your documents with semantic search
            and AI-generated answers grounded in retrieved passages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Upload and process
# --------------------------------------------------
st.subheader("1. Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    key="pdf_uploader",
    help="Upload a text-based PDF to get started.",
)

if uploaded_file is not None:
    if st.button("⚙️ Process PDF", type="primary"):
        with st.spinner(
            "Extracting text and building the search index..."
        ):
            try:
                pages = extract_pdf_text_from_bytes(
                    uploaded_file.getvalue()
                )

                chunks = chunk_pages(
                    pages,
                    chunk_size=500,
                    overlap=100,
                )

                if not chunks:
                    raise ValueError(
                        "No searchable text was found in this PDF."
                    )

                embeddings = create_embeddings(chunks)
                index = build_faiss_index(embeddings)

                st.session_state["pages"] = pages
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["filename"] = uploaded_file.name

                for key in [
                    "last_question",
                    "last_answer",
                    "last_results",
                ]:
                    st.session_state.pop(key, None)

                st.success(
                    f"Processed {len(pages)} pages into "
                    f"{len(chunks)} chunks."
                )

            except Exception as error:
                import traceback

                st.error(f"Could not process PDF: {error}")
                st.code(traceback.format_exc())

# --------------------------------------------------
# Ask questions
# --------------------------------------------------
if "index" in st.session_state:
    st.divider()
    st.subheader("2. Ask your document")

    st.caption(
        f"Active document: {st.session_state['filename']}"
    )

    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="What is the main idea of this document?",
        )

        submitted = st.form_submit_button(
            "🔍 Get Answer",
            type="primary",
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner(
                "Searching passages and generating your answer..."
            ):
                try:
                    results = search_chunks(
                        question,
                        st.session_state["chunks"],
                        st.session_state["index"],
                        top_k=3,
                    )

                    answer = generate_answer(question, results)

                    st.session_state["last_question"] = question
                    st.session_state["last_answer"] = answer
                    st.session_state["last_results"] = results

                except Exception as error:
                    import traceback

                    st.error(f"Could not generate answer: {error}")
                    st.code(traceback.format_exc())

    # --------------------------------------------------
    # Answer and citations
    # --------------------------------------------------
   
if "last_answer" in st.session_state:
    st.divider()
    st.subheader("3. Answer")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(
            f"**Question:** {st.session_state['last_question']}"
        )

    with col2:
        if st.button("✏️ New Question"):
            for key in [
                "last_question",
                "last_answer",
                "last_results",
            ]:
                st.session_state.pop(key, None)

            st.rerun()

    st.markdown(st.session_state["last_answer"])

    st.subheader("Sources and citations")

    results = st.session_state.get("last_results", [])

    if results:
        for i, result in enumerate(results, start=1):
            with st.expander(
                f"📄 Source {i} · Page {result['page']}",
                expanded=(i == 1),
            ):
                st.write(result["text"])
                st.caption(
                    f"Similarity score: {result['score']:.3f}"
                )
                st.markdown(
                    f"**Citation:** Page {result['page']}"
                )
    else:
        st.info("No source passages were retrieved.")

else:
    st.info(
        "Upload and process a PDF to start asking questions."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "AI Research Assistant · Built with Python, FAISS, "
    "Sentence Transformers, Ollama and Streamlit"
)
