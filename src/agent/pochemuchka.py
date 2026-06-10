"""
Модуль «Почемучка».

Отвечает за интеллектуальное уточнение запросов.
Если информации недостаточно для качественного ответа —
система задаёт вопросы, а не гадает.

Система специализирована на российском праве (e-commerce, селлеры
на маркетплейсах), поэтому вопрос о юрисдикции НЕ задаётся.
"""

import re

from loguru import logger


class Pochemuchka:
    """Модуль уточняющих вопросов."""

    MIN_CONTEXT_NODES = 2
    MIN_CONTEXT_DOCS = 1
    MIN_QUERY_WORDS = 4

    # Маркеры неопределённости. Сопоставление по границам слов,
    # чтобы «может ли быть расторгнут договор» не считалось неоднозначным.
    AMBIGUITY_PATTERNS = [
        r"\bкакой-нибудь\b",
        r"\bчто-нибудь\b",
        r"\bчто-то\b",
        r"\bне знаю\b",
        r"\bвроде бы\b",
        r"\bнаверное\b",
        r"\bкак-то так\b",
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
        query_lower = query.lower().strip()

        # 1. Слишком короткий запрос — не за что зацепиться
        if len(query_lower.split()) < self.MIN_QUERY_WORDS:
            questions.append(
                "Опишите ситуацию подробнее: что произошло, "
                "на какой площадке и какого результата вы хотите добиться?"
            )

        # 2. Слишком мало контекста найдено
        elif (
            len(context_nodes) < self.MIN_CONTEXT_NODES
            and len(context_docs) < self.MIN_CONTEXT_DOCS
        ):
            questions.append(
                "Я нашёл мало информации по вашему запросу. "
                "Можете уточнить: о какой площадке (Ozon, Wildberries, "
                "Яндекс Маркет) и какой ситуации идёт речь?"
            )

        # 3. Неоднозначность в запросе
        if any(re.search(p, query_lower) for p in self.AMBIGUITY_PATTERNS):
            questions.append(
                "В вашем запросе есть неопределённость. "
                "Можете описать ситуацию конкретнее?"
            )

        if questions:
            logger.info(f"Почемучка: задаю {len(questions)} уточняющих вопросов")
            return questions

        return None
