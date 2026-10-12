
import ollama


def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "I could not find the answer in the provided document."

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    print("DEBUG QUESTION:", question)
    print("DEBUG CONTEXT:", context)

    
    
    
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

Question:
{question}

Answer:
"""




    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0},
    )

    print("RAW MODEL RESPONSE:", repr(response["message"]["content"]))
    return response["message"]["content"].strip()


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
