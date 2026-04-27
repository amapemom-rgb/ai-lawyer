""" 
ИИ-Агент: ядро системы на LangGraph.

Граф состояний:
    входящий_запрос
        → почемучка (нужны ли уточнения?)
            → [да] → возврат уточняющих вопросов
            → [нет] → ретривал (поиск в базе знаний + ПГС)
                → генерация (LLM формирует ответ)
                    → валидация (проверка на галлюцинации)
                        → [не прошёл] → повторная генерация с коррекцией
                        → [прошёл] → обновление ПГС → финальный ответ

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
from src.pgs.graph import PGSGraph
from src.knowledge.store import KnowledgeStore


# === СОСТОЯНИЕ ГРАФА ===

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


# === СИСТЕМНЫЙ ПРОМПТ ===

SYSTEM_PROMPT = """Ты — ИИ-юрист, специализирующийся на российском праве в сфере e-commerce.

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


# === СОЗДАНИЕ LLM ===

def create_llm(api_key: str | None = None, model: str | None = None):
    """
    Создаёт LLM-клиент. Автоматически определяет провайдера:
    - Если есть OPENROUTER_API_KEY → OpenRouter (через OpenAI-совместимый API)
    - Если есть ANTHROPIC_API_KEY → Anthropic напрямую
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # Если передан ключ явно — определяем по префиксу
    if api_key:
        if api_key.startswith("sk-or-"):
            openrouter_key = api_key
        else:
            anthropic_key = api_key

    if openrouter_key:
        from langchain_openai import ChatOpenAI

        # На OpenRouter модели Claude указываются как anthropic/claude-...
        openrouter_model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")

        # Если передали модель без префикса провайдера — добавляем anthropic/
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


# === УЗЛЫ ГРАФА ===

def retrieve_node(state: AgentState) -> dict:
    """Узел: поиск в базе знаний и ПГС."""
    query = state["user_query"]
    logger.info(f"[Ретривал] Запрос: {query[:80]}...")

    docs = state.get("retrieved_docs", [])
    nodes = state.get("pgs_nodes", [])

    parts = []

    if nodes:
        parts.append("=== СВЯЗИ В ГРАФЕ ЗНАНИЙ ===")
        for node in nodes:
            parts.append(f"- [{node.get('type', '?')}] {node.get('title', '?')}")
            if node.get("connections"):
                for conn in node["connections"]:
                    conn_str = conn if isinstance(conn, str) else f"{conn.get('type', '?')} → {conn.get('target', '?')}"
                    parts.append(f"  → {conn_str}")

    if docs:
        parts.append("\n=== РЕЛЕВАНТНЫЕ ДОКУМЕНТЫ ===")
        for doc in docs:
            parts.append(f"\n--- {doc.get('title', 'Документ')} ---")
            parts.append(doc.get("content", "")[:2000])

    context_text = "\n".join(parts)
    logger.debug(f"[Ретривал] Контекст: {len(docs)} документов, {len(nodes)} узлов ПГС")

    return {"context_text": context_text}


def pochemuchka_node(state: AgentState) -> dict:
    """Узел: проверка — нужны ли уточнения."""
    pochemuchka = Pochemuchka()

    questions = pochemuchka.check(
        query=state["user_query"],
        context_nodes=state.get("pgs_nodes", []),
        context_docs=state.get("retrieved_docs", []),
    )

    if questions:
        logger.info(f"[Почемучка] Нужны уточнения: {len(questions)} вопросов")
        return {
            "needs_clarification": True,
            "clarification_questions": questions,
        }

    return {
        "needs_clarification": False,
        "clarification_questions": [],
    }


def generate_node(state: AgentState) -> dict:
    """Узел: генерация ответа через LLM."""
    query = state["user_query"]
    context = state.get("context_text", "")
    issues = state.get("validation_issues", [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if context:
        messages.append(HumanMessage(content=f"Контекст из базы знаний:\n{context}"))

    user_content = query
    if issues:
        user_content += (
            f"\n\n[КОРРЕКЦИЯ: предыдущий ответ содержал проблемы: "
            f"{'; '.join(issues)}. Исправь их.]"
        )

    messages.append(HumanMessage(content=user_content))

    return {"generated_response": ""}  # заполняется в LegalAgent


def validate_node(state: AgentState) -> dict:
    """Узел: валидация ответа."""
    validator = Validator()

    validation = validator.validate(
        response=state.get("generated_response", ""),
        source_documents=state.get("retrieved_docs", []),
    )

    retry_count = state.get("retry_count", 0)

    return {
        "is_valid": validation["is_valid"],
        "confidence": validation["confidence"],
        "validation_issues": validation["issues"],
        "retry_count": retry_count + (0 if validation["is_valid"] else 1),
    }


def finalize_node(state: AgentState) -> dict:
    """Узел: формирование финального результата."""
    return {
        "final_result": {
            "type": "response",
            "content": state.get("generated_response", ""),
            "sources": [doc.get("id", "") for doc in state.get("retrieved_docs", [])],
            "confidence": state.get("confidence", 0.0),
        }
    }


def clarify_node(state: AgentState) -> dict:
    """Узел: возврат уточняющих вопросов."""
    return {
        "final_result": {
            "type": "clarification",
            "questions": state.get("clarification_questions", []),
            "message": "Мне нужно уточнить несколько моментов, чтобы дать точный ответ.",
        }
    }


# === УСЛОВНЫЕ ПЕРЕХОДЫ ===

def should_clarify(state: AgentState) -> Literal["clarify", "retrieve"]:
    if state.get("needs_clarification"):
        return "clarify"
    return "retrieve"


def should_retry(state: AgentState) -> Literal["generate", "finalize"]:
    if not state.get("is_valid") and state.get("retry_count", 0) < 2:
        logger.info("[Валидация] Повторная генерация...")
        return "generate"
    return "finalize"


# === СБОРКА ГРАФА ===

def build_agent_graph() -> StateGraph:
    """
    Собирает граф агента.

    Граф:
        pochemuchka → [clarify | retrieve]
        retrieve → generate → validate → [generate | finalize]
        clarify → END
        finalize → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("pochemuchka", pochemuchka_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("clarify", clarify_node)

    graph.set_entry_point("pochemuchka")

    graph.add_conditional_edges(
        "pochemuchka",
        should_clarify,
        {"clarify": "clarify", "retrieve": "retrieve"},
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        should_retry,
        {"generate": "generate", "finalize": "finalize"},
    )

    graph.add_edge("clarify", END)
    graph.add_edge("finalize", END)

    return graph


# === ОСНОВНОЙ КЛАСС АГЕНТА ===

class LegalAgent:
    """Основной ИИ-агент юриста на LangGraph + LLM (Anthropic / OpenRouter)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        pgs: PGSGraph | None = None,
        knowledge: KnowledgeStore | None = None,
    ):
        self.llm, self.model = create_llm(api_key=api_key, model=model)
        self.pgs = pgs or PGSGraph()
        self.knowledge = knowledge or KnowledgeStore()
        self._api_key = api_key

        self._graph = build_agent_graph()
        self._compiled = self._graph.compile()

        logger.info(f"Агент LangGraph инициализирован | model={self.model}")

    def process_query(self, user_query: str) -> dict:
        """Обработка запроса пользователя через граф LangGraph."""
        logger.info(f"Новый запрос: {user_query[:100]}...")

        pgs_nodes = self.pgs.search_relevant(user_query)
        retrieved_docs = self.knowledge.search(user_query, top_k=10)

        initial_state: AgentState = {
            "user_query": user_query,
            "retrieved_docs": retrieved_docs,
            "pgs_nodes": pgs_nodes,
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

        result_state = self._run_graph(initial_state)

        final = result_state.get("final_result", {})
        if final.get("type") == "response":
            self.pgs.update_from_interaction(
                query=user_query,
                response=final.get("content", ""),
                documents_used=retrieved_docs,
            )

        return final

    def _run_graph(self, state: AgentState) -> AgentState:
        """Запуск графа с ручной обработкой узла генерации."""
        pochemuchka_result = pochemuchka_node(state)
        state.update(pochemuchka_result)

        if state["needs_clarification"]:
            clarify_result = clarify_node(state)
            state.update(clarify_result)
            return state

        retrieve_result = retrieve_node(state)
        state.update(retrieve_result)

        max_retries = 2
        for attempt in range(max_retries + 1):
            response = self._call_llm(
                query=state["user_query"],
                context=state["context_text"],
                correction=state.get("validation_issues") if attempt > 0 else None,
            )
            state["generated_response"] = response

            validate_result = validate_node(state)
            state.update(validate_result)

            if state["is_valid"] or attempt == max_retries:
                break

            logger.info(f"[Агент] Повторная генерация (попытка {attempt + 2})")

        finalize_result = finalize_node(state)
        state.update(finalize_result)

        return state

    def _call_llm(
        self, query: str, context: str, correction: list | None = None
    ) -> str:
        """Вызов LLM (Anthropic или OpenRouter)."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        if context:
            messages.append(HumanMessage(content=f"Контекст из базы знаний:\n{context}"))

        user_content = query
        if correction:
            user_content += (
                f"\n\n[КОРРЕКЦИЯ: предыдущий ответ содержал проблемы: "
                f"{'; '.join(correction)}. Исправь их.]"
            )

        messages.append(HumanMessage(content=user_content))

        response = self.llm.invoke(messages)
        return response.content

    def rotate(self) -> "LegalAgent":
        """Ротация агента: новый с чистым контекстом."""
        logger.info("Ротация агента — создаю нового с чистым контекстом")
        return LegalAgent(
            api_key=self._api_key,
            model=self.model,
            pgs=self.pgs,
            knowledge=self.knowledge,
        )
