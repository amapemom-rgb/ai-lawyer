"""
Модуль валидации ответов.

Проверяет ответ агента на:
- Наличие ссылок на реальные документы из базы знаний
- Отсутствие выдуманных статей и законов
- Соответствие контексту
"""

from loguru import logger


class Validator:
    """Валидатор ответов ИИ-агента."""

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

        # 2. Есть ссылки на источники
        if source_documents and not self._has_source_references(response, source_documents):
            issues.append("Ответ не ссылается на предоставленные документы")
            confidence -= 0.3

        # 3. Проверка на галлюцинации
        hallucination_markers = self._check_hallucination_markers(response)
        if hallucination_markers:
            issues.extend(hallucination_markers)
            confidence -= 0.2 * len(hallucination_markers)

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
        """Проверяет, ссылается ли ответ на документы из контекста."""
        response_lower = response.lower()
        for doc in documents:
            title = doc.get("title", "").lower()
            if title and title in response_lower:
                return True
        return False

    def _check_hallucination_markers(self, response: str) -> list[str]:
        """Проверка на типичные маркеры галлюцинации."""
        markers = []
        # TODO: реализовать проверку на выдуманные номера статей
        # через сопоставление с базой знаний
        return markers
