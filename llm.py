
import os
from groq import Groq


def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "I could not find the answer in the provided document."

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using the document passages below.

Instructions:
- First, look for a direct answer in the passages.
- If a passage directly answers the question, state that answer clearly.
- Do not reject an answer merely because the passage uses different wording.
- Do not use outside knowledge.
- If the passages do not contain the answer, reply exactly:
  I could not find the answer in the provided document.
- Cite the page number for supported answers.

Document passages:
{context}

Question: {question}

Answer:
"""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Configure it in Streamlit Cloud Secrets."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "I could not generate an answer."

    return answer.strip()


if __name__ == "__main__":
    question = "What does the old lighthouse keeper remember?"

    retrieved_chunks = [
        {
            "page": 1,
            "text": (
                "The old lighthouse keeper remembered "
                "the last warm day of autumn and nobody ever knew."
            ),
        }
    ]

    print(generate_answer(question, retrieved_chunks))
