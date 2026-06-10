"""
Тесты модуля «Почемучка».
"""

from src.agent.pochemuchka import Pochemuchka

CONTEXT_NODES = [{"type": "Law", "title": "Test"}] * 3
CONTEXT_DOCS = [{"title": "Test"}]


def test_asks_clarification_when_no_context():
    p = Pochemuchka()
    result = p.check(
        query="Помогите разобраться с возвратом товара покупателю",
        context_nodes=[],
        context_docs=[],
    )
    assert result is not None
    assert len(result) > 0


def test_asks_details_for_too_short_query():
    p = Pochemuchka()
    result = p.check(
        query="Помогите с документом",
        context_nodes=CONTEXT_NODES,
        context_docs=CONTEXT_DOCS,
    )
    assert result is not None


def test_no_questions_when_context_sufficient():
    p = Pochemuchka()
    result = p.check(
        query="Статья 18 закона о защите прав потребителей РФ",
        context_nodes=CONTEXT_NODES,
        context_docs=CONTEXT_DOCS,
    )
    assert result is None


def test_detects_ambiguity():
    p = Pochemuchka()
    result = p.check(
        query="Наверное что-то подойдёт из законов про возвраты",
        context_nodes=CONTEXT_NODES,
        context_docs=CONTEXT_DOCS,
    )
    assert result is not None


def test_no_false_positive_on_mozhet_li_byt():
    """«Может ли быть...» — легитимный юридический вопрос, не неопределённость."""
    p = Pochemuchka()
    result = p.check(
        query="Может ли быть расторгнут договор поставки в одностороннем порядке?",
        context_nodes=CONTEXT_NODES,
        context_docs=CONTEXT_DOCS,
    )
    assert result is None


def test_no_jurisdiction_question():
    """Система специализирована на РФ — вопрос о стране не задаётся."""
    p = Pochemuchka()
    result = p.check(
        query="Какой закон регулирует возврат товаров на маркетплейсе?",
        context_nodes=CONTEXT_NODES,
        context_docs=CONTEXT_DOCS,
    )
    if result:
        assert not any("стран" in q or "юрисдикц" in q for q in result)
