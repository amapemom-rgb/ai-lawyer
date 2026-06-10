"""
Модуль валидации ответов.

Проверяет ответ агента на:
- Наличие ссылок на реальные документы из базы знаний (нечёткое сопоставление)
- Отсутствие выдуманных статей (номера статей в ответе сверяются с источниками)
- Минимальную содержательность
"""

import re

from loguru import logger

# «статья 18», «ст. 18», «ст.18.1», «статьи 28»
_ARTICLE_RE = re.compile(
    r"\bстат(?:ья|ьи|ье|ьёй|ей|ью)\s+(\d+(?:\.\d+)*)|\bст\.?\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Слова, не несущие смысла при сопоставлении названий
_STOPWORDS = {
    "о", "об", "обо", "и", "в", "во", "на", "по", "от", "для", "при", "с", "со",
    "к", "за", "из", "не", "до", "как", "что", "the", "of", "a", "an",
}


def _extract_articles(text: str) -> set[str]:
    """Извлекает номера статей из текста."""
    articles = set()
    for m in _ARTICLE_RE.finditer(text):
        num = m.group(1) or m.group(2)
        if num:
            articles.add(num)
    return articles


def _significant_tokens(text: str) -> set[str]:
    """Значимые слова (≥4 символов, не стоп-слова) в нижнем регистре."""
    tokens = re.findall(r"[а-яёa-z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}


class Validator:
    """Валидатор ответов ИИ-агента."""

    # Доля значимых слов названия документа, которая должна
    # встретиться в ответе, чтобы считать ссылку состоявшейся
    TITLE_OVERLAP_THRESHOLD = 0.5

    def validate(self, response: str, source_documents: list) -> dict:
        """
        Валидация ответа агента.

        Returns:
            {
                "is_valid": bool,
                "confidence": float,  # 0.0 - 1.0
                "issues": list[str],
            }
        """
        issues = []
        confidence = 1.0

        # 1. Ответ не пустой
        if not response or len(response.strip()) < 20:
            issues.append("Ответ слишком короткий")
            confidence -= 0.5

        # 2. Есть ссылки на источники (нечёткое сопоставление)
        if source_documents and response and not self._has_source_references(response, source_documents):
            issues.append("Ответ не ссылается на предоставленные документы")
            confidence -= 0.3

        # 3. Проверка на выдуманные статьи
        hallucinated = self._check_hallucinated_articles(response, source_documents)
        if hallucinated:
            issues.append(
                "Возможно выдуманные номера статей (нет в источниках): "
                + ", ".join(sorted(hallucinated))
            )
            confidence -= 0.2 * len(hallucinated)

        confidence = max(0.0, min(1.0, confidence))

        result = {
            "is_valid": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
        }

        if issues:
            logger.warning(f"Валидация: {issues}")
        else:
            logger.info(f"Валидация пройдена | confidence={confidence:.2f}")

        return result

    def _has_source_references(self, response: str, documents: list) -> bool:
        """
        Нечёткая проверка, ссылается ли ответ на документы из контекста.

        Считаем ссылку состоявшейся, если выполнено любое из:
        - точное вхождение названия документа;
        - ≥50% значимых слов названия встречаются в ответе;
        - в ответе упомянута статья, которая есть в тексте источника.
        """
        response_lower = response.lower()
        response_tokens = _significant_tokens(response)
        response_articles = _extract_articles(response)

        for doc in documents:
            title = (doc.get("title") or "").lower()
            if title and title in response_lower:
                return True

            title_tokens = _significant_tokens(title)
            if title_tokens:
                overlap = len(title_tokens & response_tokens) / len(title_tokens)
                if overlap >= self.TITLE_OVERLAP_THRESHOLD:
                    return True

            if response_articles:
                doc_articles = _extract_articles(doc.get("content") or "")
                if response_articles & doc_articles:
                    return True

        return False

    def _check_hallucinated_articles(self, response: str, documents: list) -> set[str]:
        """
        Номера статей, упомянутые в ответе, но отсутствующие в источниках.

        Если источников нет — проверка пропускается (нечего сверять).
        """
        if not documents or not response:
            return set()

        response_articles = _extract_articles(response)
        if not response_articles:
            return set()

        source_articles: set[str] = set()
        for doc in documents:
            source_articles |= _extract_articles(doc.get("content") or "")
            source_articles |= _extract_articles(doc.get("title") or "")

        return response_articles - source_articles
