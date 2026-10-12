
from llm import generate_answer


def test_generate_answer_uses_document_context(monkeypatch):
    expected = "The keeper remembers autumn. [Page 1]"

    def fake_chat(**kwargs):
        prompt = kwargs["messages"][0]["content"]

        assert "The lighthouse keeper remembered autumn." in prompt
        assert "What does the keeper remember?" in prompt

        return {
            "message": {
                "content": expected
            }
        }

    monkeypatch.setattr("llm.ollama.chat", fake_chat)

    chunks = [
        {
            "page": 1,
            "text": "The lighthouse keeper remembered autumn.",
            "score": 0.8,
        }
    ]

    answer = generate_answer(
        "What does the keeper remember?",
        chunks,
    )

    assert answer == expected
