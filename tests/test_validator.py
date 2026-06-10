"""
Тесты модуля валидации.
"""

from src.agent.validator import Validator

DOCS = [
    {
        "id": "doc1",
        "title": "Закон о защите прав потребителей",
        "content": "Статья 18. Права потребителя при обнаружении в товаре недостатков... "
                   "Статья 25. Право потребителя на обмен товара надлежащего качества...",
    },
    {
        "id": "doc2",
        "title": "Правила возврата Ozon",
        "content": "Возврат товара продавцу осуществляется в течение 30 дней...",
    },
]


def test_valid_response_with_exact_title():
    v = Validator()
    result = v.validate(
        response="Согласно Закону о защите прав потребителей, вы вправе вернуть товар.",
        source_documents=DOCS,
    )
    assert result["is_valid"]
    assert result["confidence"] == 1.0


def test_valid_response_with_article_reference():
    """Ссылка на статью из источника засчитывается, даже без точного названия документа."""
    v = Validator()
    result = v.validate(
        response="По ст. 18 ЗоЗПП покупатель вправе предъявить требования при недостатках товара. "
                 "Рекомендую зафиксировать дефект и ответить на претензию в установленный срок.",
        source_documents=DOCS,
    )
    assert result["is_valid"], result["issues"]


def test_detects_hallucinated_article():
    v = Validator()
    result = v.validate(
        response="Согласно статье 999 Закона о защите прав потребителей, продавец обязан вернуть деньги.",
        source_documents=DOCS,
    )
    assert not result["is_valid"]
    assert any("999" in issue for issue in result["issues"])


def test_short_response_invalid():
    v = Validator()
    result = v.validate(response="Да.", source_documents=DOCS)
    assert not result["is_valid"]


def test_no_sources_skips_reference_check():
    v = Validator()
    result = v.validate(
        response="К сожалению, в моей базе знаний нет информации по этому вопросу. "
                 "Рекомендую обратиться к юристу очно.",
        source_documents=[],
    )
    assert result["is_valid"]


def test_response_without_any_reference_flagged():
    v = Validator()
    result = v.validate(
        response="Просто верните товар и всё будет хорошо, никаких проблем не возникнет.",
        source_documents=DOCS,
    )
    assert not result["is_valid"]
    assert any("не ссылается" in issue for issue in result["issues"])
