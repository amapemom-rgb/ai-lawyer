"""
Тесты модуля «Почемучка».
"""

from src.agent.pochemuchka import Pochemuchka


def test_asks_clarification_when_no_context():
    p = Pochemuchka()
    result = p.check(
        query="Помогите с документом",
        context_nodes=[],
        context_docs=[],
    )
    assert result is not None
    assert len(result) > 0


def test_asks_about_jurisdiction():
    p = Pochemuchka()
    result = p.check(
        query="Какой закон регулирует возврат товаров?",
        context_nodes=[{"type": "Law", "title": "Test"}] * 3,
        context_docs=[{"title": "Test"}],
    )
    assert result is not None
    assert any("стран" in q or "регион" in q for q in result)


def test_no_questions_when_context_sufficient():
    p = Pochemuchka()
    result = p.check(
        query="Статья 18 закона о защите прав потребителей РФ",
        context_nodes=[{"type": "Law", "title": "Test"}] * 3,
        context_docs=[{"title": "Test"}],
    )
    assert result is None


def test_detects_ambiguity():
    p = Pochemuchka()
    result = p.check(
        query="Может быть что-то подойдёт из законов",
        context_nodes=[{"type": "Law", "title": "Test"}] * 3,
        context_docs=[{"title": "Test"}],
    )
    assert result is not None
