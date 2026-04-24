"""
ИИ-Агент: ядро системы.

Сменяемый агент с чистым контекстным окном.
Всё состояние хранится в ПГС и базах — агент stateless.
"""

import anthropic
from loguru import logger
from src.agent.pochemuchka import Pochemuchka
from src.agent.validator import Validator
from src.pgs.graph import PGSGraph
from src.knowledge.store import KnowledgeStore


class LegalAgent:
    """Основной ИИ-агент юриста."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        pgs: PGSGraph | None = None,
        knowledge: KnowledgeStore | None = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.pgs = pgs or PGSGraph()
        self.knowledge = knowledge or KnowledgeStore()
        self.pochemuchka = Pochemuchka()
        self.validator = Validator()
        self.conversation_history: list[dict] = []
        logger.info(f"Агент инициализирован | model={model}")

    def process_query(self, user_query: str) -> dict:
        """
        Обработка запроса пользователя.

        1. Поиск релевантных узлов в ПГС
        2. Извлечение документов из базы знаний
        3. Формирование контекста и запрос к LLM
        4. Валидация ответа
        5. Обновление ПГС новыми связями
        """
        logger.info(f"Новый запрос: {user_query[:100]}...")

        # 1. Поиск в ПГС
        relevant_nodes = self.pgs.search_relevant(user_query)
        logger.debug(f"Найдено узлов в ПГС: {len(relevant_nodes)}")

        # 2. Извлечение из базы знаний
        documents = self.knowledge.search(user_query, top_k=10)
        logger.debug(f"Найдено документов: {len(documents)}")

        # 3. Проверка: достаточно ли информации?
        clarification = self.pochemuchka.check(
            query=user_query,
            context_nodes=relevant_nodes,
            context_docs=documents,
        )
        if clarification:
            return {
                "type": "clarification",
                "questions": clarification,
                "message": "Мне нужно уточнить несколько моментов, чтобы дать точный ответ.",
            }

        # 4. Генерация ответа
        context = self._build_context(relevant_nodes, documents)
        response = self._call_llm(user_query, context)

        # 5. Валидация
        validation = self.validator.validate(response, documents)
        if not validation["is_valid"]:
            logger.warning(f"Валидация не пройдена: {validation['issues']}")
            response = self._call_llm(
                user_query, context, correction=validation["issues"]
            )

        # 6. Обновление ПГС
        self.pgs.update_from_interaction(
            query=user_query,
            response=response,
            documents_used=documents,
        )

        return {
            "type": "response",
            "content": response,
            "sources": [doc["id"] for doc in documents],
            "confidence": validation.get("confidence", 0.0),
        }

    def _build_context(self, nodes: list, documents: list) -> str:
        """Формирование контекста из узлов ПГС и документов."""
        parts = []

        if nodes:
            parts.append("=== СВЯЗИ В ГРАФЕ ЗНАНИЙ ===")
            for node in nodes:
                parts.append(f"- [{node.get('type', '?')}] {node.get('title', '?')}")
                if node.get("connections"):
                    for conn in node["connections"]:
                        parts.append(f"  → {conn}")

        if documents:
            parts.append("\n=== РЕЛЕВАНТНЫЕ ДОКУМЕНТЫ ===")
            for doc in documents:
                parts.append(f"\n--- {doc.get('title', 'Документ')} ---")
                parts.append(doc.get("content", "")[:2000])

        return "\n".join(parts)

    def _call_llm(
        self, query: str, context: str, correction: list | None = None
    ) -> str:
        """Вызов Claude API."""
        system_prompt = (
            "Ты — ИИ-юрист. Отвечай точно, ссылайся на конкретные документы и статьи. "
            "Если информации недостаточно — скажи об этом прямо. "
            "Не выдумывай законы и статьи."
        )

        messages = []

        if context:
            messages.append({"role": "user", "content": f"Контекст:\n{context}"})
            messages.append({"role": "assistant", "content": "Принял контекст. Готов ответить на вопрос."})

        user_content = query
        if correction:
            user_content += f"\n\n[КОРРЕКЦИЯ: предыдущий ответ содержал проблемы: {'; '.join(correction)}. Исправь их.]"

        messages.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )

        return response.content[0].text

    def rotate(self) -> "LegalAgent":
        """
        Ротация агента: создание нового агента с чистым контекстом.
        ПГС и база знаний переиспользуются.
        """
        logger.info("Ротация агента — создаю нового с чистым контекстом")
        return LegalAgent(
            api_key=self.client.api_key,
            model=self.model,
            pgs=self.pgs,
            knowledge=self.knowledge,
        )
