"""
Модуль «Почемучка».

Отвечает за интеллектуальное уточнение запросов.
Если информации недостаточно для качественного ответа —
система задаёт вопросы, а не гадает.

Каждый ответ пользователя на уточняющий вопрос
становится новым знанием в ПГС.
"""

from loguru import logger


class Pochemuchka:
    """Модуль уточняющих вопросов."""

    MIN_CONTEXT_NODES = 2
    MIN_CONTEXT_DOCS = 1
    AMBIGUITY_KEYWORDS = [
        "любой", "какой-нибудь", "что-то", "наверное",
        "может быть", "не знаю", "вроде", "примерно",
    ]

    def check(
        self,
        query: str,
        context_nodes: list,
        context_docs: list,
    ) -> list[str] | None:
        """
        Проверяет, нужно ли задать уточняющие вопросы.

        Returns:
            Список вопросов или None, если уточнение не требуется.
        """
        questions = []

        # 1. Слишком мало контекста
        if len(context_nodes) < self.MIN_CONTEXT_NODES and len(context_docs) < self.MIN_CONTEXT_DOCS:
            questions.append(
                "Я нашёл мало информации по вашему запросу. "
                "Можете уточнить: к какой области права относится ваш вопрос?"
            )

        # 2. Неоднозначность в запросе
        query_lower = query.lower()
        if any(kw in query_lower for kw in self.AMBIGUITY_KEYWORDS):
            questions.append(
                "В вашем запросе есть неопределённость. "
                "Можете описать ситуацию конкретнее?"
            )

        # 3. Нет указания на юрисдикцию
        if not self._has_jurisdiction_hint(query):
            questions.append(
                "Для какой страны/региона нужна юридическая информация?"
            )

        if questions:
            logger.info(f"Почемучка: задаю {len(questions)} уточняющих вопросов")
            return questions

        return None

    def _has_jurisdiction_hint(self, query: str) -> bool:
        """Проверяет, упоминается ли юрисдикция в запросе."""
        jurisdiction_hints = [
            "рф", "россия", "российск", "федеральн",
            "гк ", "тк ", "ук ", "коап", "гпк", "апк",
            "казахстан", "беларус", "украин",
        ]
        query_lower = query.lower()
        return any(hint in query_lower for hint in jurisdiction_hints)
