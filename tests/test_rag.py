import httpx
import pytest

from garage_copilot.config import Settings
from garage_copilot.llm import OpenAIClient
from garage_copilot.rag import RagAssistant, chunk_document, load_directory
from garage_copilot.rag.assistant import NO_ANSWER, expand_query
from garage_copilot.rag.chunking import Document


@pytest.fixture(scope="module")
def rag() -> RagAssistant:
    return RagAssistant.from_directory(Settings().knowledge_dir)


def test_chunker_splits_on_headings_and_long_sections():
    body = "# Title\n\n## A\n\none\n\n## B\n\n" + "\n\n".join(["x" * 300] * 4)
    chunks = chunk_document(Document("doc.md", "Doc", "test", body), max_chars=700)
    headings = [c.heading for c in chunks]
    assert headings.count("A") == 1
    assert headings.count("B") == 2  # 4 x 300 chars packed into two chunks
    assert all(len(c.text) <= 700 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_knowledge_base_loads_front_matter():
    chunks = load_directory(Settings().knowledge_dir)
    assert {c.doc_title for c in chunks} >= {"Courtesy car terms", "MOT and servicing guide"}
    assert all("illustrative" in c.source for c in chunks)


@pytest.mark.parametrize(
    ("question", "expected_citation", "expected_text"),
    [
        ("What is the excess on the courtesy car?", "Damage and excess", "£250"),
        ("When does a car need its first MOT?", "When an MOT is due", "three years old"),
        (
            "What happens if I get a parking fine in the loan car?",
            "Fines and penalty charges",
            "hirer",
        ),
        (
            "Will a new stereo work in a Golf with Dynaudio?",
            "Volkswagen Golf",
            "amplifier interface",
        ),
    ],
)
def test_retrieval_finds_the_right_section(rag, question, expected_citation, expected_text):
    answer = rag.ask(question)
    assert any(expected_citation in c for c in answer.citations)
    assert expected_text in answer.text
    assert "[1]" in answer.text


def test_out_of_scope_question_is_not_answered(rag):
    answer = rag.ask("What is the capital of France?")
    assert answer.text == NO_ANSWER
    assert not answer.grounded


def test_query_expansion():
    assert "courtesy car" in expand_query("Can I keep the loan car?")
    assert expand_query("MOT due?") == "MOT due?"


def test_llm_receives_numbered_context(rag):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "The excess is £250 [1]."}}]}
        )

    llm = OpenAIClient("key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    answer = RagAssistant(rag.store, llm=llm).ask("What is the courtesy car excess?")
    assert answer.text == "The excess is £250 [1]."
    assert "[1] (Courtesy car terms" in seen["body"]


def test_retrieval_quality_does_not_regress(rag):
    from garage_copilot.config import DATA_DIR
    from garage_copilot.rag.evaluate import evaluate_retrieval, load_questions

    report = evaluate_retrieval(
        rag.store, load_questions(DATA_DIR / "eval" / "rag_questions.jsonl")
    )
    assert report.hit_at_k >= 0.9, report.misses
    assert report.mrr >= 0.8
