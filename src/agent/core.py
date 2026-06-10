"""
ИИ-Агент: ядро системы на LangGraph.

Граф состояний (реально исполняется через compiled.invoke()):

    retrieve (поиск в ПГС + базе знаний)
        → pochemuchka (нужны ли уточнения?)
            → [да] → clarify → END
            → [нет] → generate (LLM)
                → validate (проверка ответа)
                    → [не прошёл, retry < 2] → generate (с коррекцией)
                    → [прошёл / лимит] → finalize → END

Поддерживаемые провайдеры:
- Anthropic (напрямую): ANTHROPIC_API_KEY
- OpenRouter (Claude и другие модели): OPENROUTER_API_KEY
"""

from __future__ import annotations

import os
from typing import TypedDict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from loguru import logger

from src.agent.pochemuchka import Pochemuchka
from src.agent.validator import Validator


MAX_RETRIES = 2


class AgentState(TypedDict):
    """Полное состояние одного цикла обработки запроса."""
    user_query: str
    retrieved_docs: list[dict]
    pgs_nodes: list[dict]
    needs_clarification: bool
    clarification_questions: list[str]
    generated_response: str
    context_text: str
    is_valid: bool
    validation_issues: list[str]
    confidence: float
    retry_count: int
    final_result: dict


SYSTEM_PROMPT = """Ты — ИИ-юрист для продавцов (селлеров) на российских маркетплейсах \
(Ozon, Wildberries, Яндекс Маркет). Специализация: российское право в сфере e-commerce, \
правила площадок, тарифы, комиссии, споры с маркетплейсом, блокировки, претензии покупателей.

ПРАВИЛА:
1. Отвечай точно, ссылайся на конкретные статьи законов и документы из контекста.
2. Если информации в контексте недостаточно — скажи об этом прямо.
3. НИКОГДА не выдумывай номера статей, законов или нормативных актов.
4. Если вопрос выходит за рамки предоставленного контекста — так и скажи.
5. Используй простой, понятный язык. Юридические термины объясняй.
6. Структурируй ответ: сначала прямой ответ, потом обоснование со ссылками.

ФОРМАТ ОТВЕТА:
- Прямой ответ на вопрос (1-2 предложения)
- Обоснование со ссылками на конкретные статьи/документы
- Практический совет (что делать дальше)
"""


def create_llm(api_key: str | None = None, model: str | None = None):
    """
    Создаёт LLM-клиент. Автоматически определяет провайдера:
    - Если есть OPENROUTER_API_KEY → OpenRouter (через OpenAI-совместимый API)
    - Если есть ANTHROPIC_API_KEY → Anthropic напрямую
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key:
        if api_key.startswith("sk-or-"):
            openrouter_key = api_key
        else:
            anthropic_key = api_key

    if openrouter_key:
        from langchain_openai import ChatOpenAI

        openrouter_model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")
        if "/" not in openrouter_model:
            openrouter_model = f"anthropic/{openrouter_model}"

        llm = ChatOpenAI(
            model=openrouter_model,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=4096,
            default_headers={
                "HTTP-Referer": "https://github.com/amapemom-rgb/ai-lawyer",
                "X-Title": "AI-Lawyer",
            },
        )
        logger.info(f"LLM: OpenRouter | model={openrouter_model}")
        return llm, openrouter_model

    elif anthropic_key:
        from langchain_anthropic import ChatAnthropic

        anthropic_model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        llm = ChatAnthropic(
            model=anthropic_model,
            anthropic_api_key=anthropic_key,
            max_tokens=4096,
        )
        logger.info(f"LLM: Anthropic | model={anthropic_model}")
        return llm, anthropic_model

    else:
        raise ValueError(
            "API-ключ не найден! Укажите один из:\n"
            "  OPENROUTER_API_KEY=sk-or-...  (OpenRouter)\n"
            "  ANTHROPIC_API_KEY=sk-ant-...   (Anthropic напрямую)\n"
            "в файле .env или переменных окружения."
        )


class LegalAgent:
    """ИИ-агент юриста. Весь цикл обработки исполняется графом LangGraph."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        pgs=None,
        knowledge=None,
        llm=None,
    ):
        if llm is not None:
            # Инъекция готового LLM (тесты, кастомные клиенты)
            self.llm, self.model = llm, model or "injected"
        else:
            self.llm, self.model = create_llm(api_key=api_key, model=model)

        # Ленивая инициализация зависимостей — тяжёлые импорты только при надобности
        if pgs is None:
            from src.pgs.graph import PGSGraph
            pgs = PGSGraph()
        if knowledge is None:
            from src.knowledge.store import KnowledgeStore
            knowledge = KnowledgeStore()

        self.pgs = pgs
        self.knowledge = knowledge
        self._api_key = api_key

        self.pochemuchka = Pochemuchka()
        self.validator = Validator()

        self._compiled = self._build_graph().compile()
        logger.info(f"Агент LangGraph инициализирован | model={self.model}")

    # --- УЗЛЫ ГРАФА ---

    def _retrieve_node(self, state: AgentState) -> dict:
        """Поиск в ПГС и базе знаний, сборка текстового контекста."""
        query = state["user_query"]
        logger.info(f"[Ретривал] Запрос: {query[:80]}...")

        nodes = self.pgs.search_relevant(query)
        docs = self.knowledge.search(query, top_k=10)

        parts = []
        if nodes:
            parts.append("=== СВЯЗИ В ГРАФЕ ЗНАНИЙ ===")
            for node in nodes:
                parts.append(f"- [{node.get('type', '?')}] {node.get('title', '?')}")
                for conn in node.get("connections") or []:
                    conn_str = conn if isinstance(conn, str) else f"{conn.get('type', '?')} → {conn.get('target', '?')}"
                    parts.append(f"  → {conn_str}")

        if docs:
            parts.append("\n=== РЕЛЕВАНТНЫЕ ДОКУМЕНТЫ ===")
            for doc in docs:
                parts.append(f"\n--- {doc.get('title', 'Документ')} ---")
                parts.append(doc.get("content", "")[:2000])

        logger.debug(f"[Ретривал] Контекст: {len(docs)} документов, {len(nodes)} узлов ПГС")
        return {
            "retrieved_docs": docs,
            "pgs_nodes": nodes,
            "context_text": "\n".join(parts),
        }

    def _pochemuchka_node(self, state: AgentState) -> dict:
        """Проверка — нужны ли уточнения."""
        questions = self.pochemuchka.check(
            query=state["user_query"],
            context_nodes=state.get("pgs_nodes", []),
            context_docs=state.get("retrieved_docs", []),
        )
        if questions:
            logger.info(f"[Почемучка] Нужны уточнения: {len(questions)} вопросов")
            return {"needs_clarification": True, "clarification_questions": questions}
        return {"needs_clarification": False, "clarification_questions": []}

    def _generate_node(self, state: AgentState) -> dict:
        """Генерация ответа через LLM (с коррекцией при повторе)."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        context = state.get("context_text", "")
        if context:
            messages.append(HumanMessage(content=f"Контекст из базы знаний:\n{context}"))

        user_content = state["user_query"]
        issues = state.get("validation_issues", [])
        if state.get("retry_count", 0) > 0 and issues:
            user_content += (
                f"\n\n[КОРРЕКЦИЯ: предыдущий ответ содержал проблемы: "
                f"{'; '.join(issues)}. Исправь их.]"
            )
        messages.append(HumanMessage(content=user_content))

        response = self.llm.invoke(messages)
        return {"generated_response": response.content}

    def _validate_node(self, state: AgentState) -> dict:
        """Валидация ответа. Единственное место, где растёт retry_count."""
        validation = self.validator.validate(
            response=state.get("generated_response", ""),
            source_documents=state.get("retrieved_docs", []),
        )
        retry_count = state.get("retry_count", 0)
        if not validation["is_valid"]:
            retry_count += 1

        return {
            "is_valid": validation["is_valid"],
            "confidence": validation["confidence"],
            "validation_issues": validation["issues"],
            "retry_count": retry_count,
        }

    def _finalize_node(self, state: AgentState) -> dict:
        return {
            "final_result": {
                "type": "response",
                "content": state.get("generated_response", ""),
                "sources": [doc.get("id", "") for doc in state.get("retrieved_docs", [])],
                "confidence": state.get("confidence", 0.0),
            }
        }

    def _clarify_node(self, state: AgentState) -> dict:
        return {
            "final_result": {
                "type": "clarification",
                "questions": state.get("clarification_questions", []),
                "message": "Мне нужно уточнить несколько моментов, чтобы дать точный ответ.",
            }
        }

    # --- УСЛОВНЫЕ ПЕРЕХОДЫ ---

    @staticmethod
    def _should_clarify(state: AgentState) -> Literal["clarify", "generate"]:
        return "clarify" if state.get("needs_clarification") else "generate"

    @staticmethod
    def _should_retry(state: AgentState) -> Literal["generate", "finalize"]:
        if not state.get("is_valid") and state.get("retry_count", 0) <= MAX_RETRIES:
            logger.info("[Валидация] Повторная генерация...")
            return "generate"
        return "finalize"

    # --- СБОРКА ГРАФА ---

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("pochemuchka", self._pochemuchka_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_node("clarify", self._clarify_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "pochemuchka")
        graph.add_conditional_edges(
            "pochemuchka", self._should_clarify,
            {"clarify": "clarify", "generate": "generate"},
        )
        graph.add_edge("generate", "validate")
        graph.add_conditional_edges(
            "validate", self._should_retry,
            {"generate": "generate", "finalize": "finalize"},
        )
        graph.add_edge("clarify", END)
        graph.add_edge("finalize", END)

        return graph

    # --- ПУБЛИЧНЫЙ API ---

    def process_query(self, user_query: str) -> dict:
        """Обработка запроса пользователя: один вызов скомпилированного графа."""
        logger.info(f"Новый запрос: {user_query[:100]}...")

        initial_state: AgentState = {
            "user_query": user_query,
            "retrieved_docs": [],
            "pgs_nodes": [],
            "needs_clarification": False,
            "clarification_questions": [],
            "generated_response": "",
            "context_text": "",
            "is_valid": False,
            "validation_issues": [],
            "confidence": 0.0,
            "retry_count": 0,
            "final_result": {},
        }

        result_state = self._compiled.invoke(initial_state)
        final = result_state.get("final_result", {})

        if final.get("type") == "response":
            self.pgs.update_from_interaction(
                query=user_query,
                response=final.get("content", ""),
                documents_used=result_state.get("retrieved_docs", []),
            )

        return final

    def rotate(self) -> "LegalAgent":
        """Ротация агента: новый с чистым контекстом."""
        logger.info("Ротация агента — создаю нового с чистым контекстом")
        return LegalAgent(
            api_key=self._api_key,
            model=self.model,
            pgs=self.pgs,
            knowledge=self.knowledge,
            llm=self.llm,
        )
