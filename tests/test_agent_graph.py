"""
Интеграционные тесты графа LangGraph с мок-LLM и мок-хранилищами.

Проверяют, что цикл retrieve → pochemuchka → generate → validate → finalize
реально исполняется скомпилированным графом.
"""

import pytest

langgraph = pytest.importorskip("langgraph")

from src.agent.core import LegalAgent


class FakeLLM:
    """Мок-LLM: возвращает заданные ответы по очереди."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        text = self.responses.pop(0) if self.responses else "Ответ по умолчанию."

        class R:
            content = text
        return R()


class FakePGS:
    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.interactions = []

    def search_relevant(self, query, limit=20):
        return self.nodes

    def update_from_interaction(self, query, response, documents_used):
        self.interactions.append(query)


class FakeKnowledge:
    def __init__(self, docs=None):
        self.docs = docs or []

    def search(self, query, top_k=10):
        return self.docs


GOOD_DOCS = [
    {
        "id": "doc1",
        "title": "Закон о защите прав потребителей",
        "content": "Статья 18. Права потребителя при обнаружении недостатков...",
    },
    {"id": "doc2", "title": "Правила Ozon", "content": "Возврат в течение 30 дней..."},
]
GOOD_NODES = [{"type": "Law", "title": "ЗоЗПП", "connections": []}, {"type": "Law", "title": "ГК РФ", "connections": []}]

GOOD_RESPONSE = (
    "Согласно статье 18 Закона о защите прав потребителей, покупатель вправе "
    "предъявить претензию при недостатках товара. Рекомендую ответить на претензию письменно."
)


def make_agent(llm, docs=GOOD_DOCS, nodes=GOOD_NODES):
    return LegalAgent(
        llm=llm,
        model="fake",
        pgs=FakePGS(nodes),
        knowledge=FakeKnowledge(docs),
    )


def test_full_cycle_returns_response():
    llm = FakeLLM([GOOD_RESPONSE])
    agent = make_agent(llm)

    result = agent.process_query("Покупатель требует возврат товара с недостатком, что делать продавцу?")

    assert result["type"] == "response"
    assert "статье 18" in result["content"]
    assert result["sources"] == ["doc1", "doc2"]
    assert result["confidence"] == 1.0
    assert llm.calls == 1
    # ПГС обновлена после успешного ответа
    assert agent.pgs.interactions


def test_clarification_short_circuits_llm():
    """При пустом контексте почемучка возвращает вопросы, LLM не вызывается."""
    llm = FakeLLM([GOOD_RESPONSE])
    agent = make_agent(llm, docs=[], nodes=[])

    result = agent.process_query("Меня заблокировали")

    assert result["type"] == "clarification"
    assert result["questions"]
    assert llm.calls == 0
    assert not agent.pgs.interactions


def test_retry_on_invalid_then_success():
    """Первый ответ с выдуманной статьёй → ретрай → валидный ответ."""
    bad = "Согласно статье 999 Закона о защите прав потребителей, всё можно."
    llm = FakeLLM([bad, GOOD_RESPONSE])
    agent = make_agent(llm)

    result = agent.process_query("Покупатель требует возврат товара с недостатком, что делать продавцу?")

    assert result["type"] == "response"
    assert llm.calls == 2
    assert "статье 18" in result["content"]


def test_retry_limit_respected():
    """Все ответы невалидны → не больше 1 + MAX_RETRIES вызовов LLM."""
    bad = "Согласно статье 999, всё можно."
    llm = FakeLLM([bad] * 10)
    agent = make_agent(llm)

    result = agent.process_query("Покупатель требует возврат товара с недостатком, что делать продавцу?")

    assert result["type"] == "response"
    assert llm.calls == 3  # 1 попытка + 2 ретрая
    assert result["confidence"] < 1.0
