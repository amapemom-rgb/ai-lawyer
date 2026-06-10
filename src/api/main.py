"""
FastAPI — REST API для ИИ-юриста.

Эндпоинты:
- GET  /health          — проверка состояния
- POST /query           — запрос к ИИ-юристу (LangGraph)
- POST /documents/upload — загрузка документа в базу знаний
- POST /documents/search — поиск по базе знаний

Защита: если задана переменная окружения API_KEY, все эндпоинты
(кроме /health) требуют заголовок X-API-Key.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html"}

# Глобальные объекты
agent = None
knowledge = None
pgs = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и остановка компонентов."""
    global agent, knowledge, pgs

    logger.info("AI Lawyer запускается (LangGraph + LlamaIndex)...")

    from src.pgs.graph import PGSGraph
    from src.knowledge.store import KnowledgeStore
    from src.agent.core import LegalAgent

    # ПГС
    neo4j_uri = os.getenv("NEO4J_URI")
    pgs = PGSGraph(
        uri=neo4j_uri,
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    ) if neo4j_uri else PGSGraph()
    pgs.ensure_indexes()

    # База знаний (LlamaIndex + ChromaDB)
    chroma_host = os.getenv("CHROMA_HOST")
    knowledge = KnowledgeStore(
        host=chroma_host,
        port=int(os.getenv("CHROMA_PORT", "8000")),
    ) if chroma_host else KnowledgeStore()

    # Агент (LangGraph): Anthropic напрямую или OpenRouter
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if api_key:
        agent = LegalAgent(
            api_key=api_key,
            pgs=pgs,
            knowledge=knowledge,
        )
        logger.info("Агент LangGraph инициализирован")
    else:
        logger.warning("Ни ANTHROPIC_API_KEY, ни OPENROUTER_API_KEY не заданы — агент не запущен")

    yield

    if pgs:
        pgs.close()
    logger.info("AI Lawyer остановлен")


async def verify_api_key(x_api_key: str | None = Header(default=None)):
    """Простая защита по ключу. Активна, только если задан API_KEY."""
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


app = FastAPI(
    title="AI Lawyer — ИИ-Юрист",
    description="Юридическая система: LangGraph (оркестрация) + LlamaIndex (RAG) + Claude (генерация)",
    version="0.3.0",
    lifespan=lifespan,
)


# === Модели запросов ===

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# === Эндпоинты ===

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.3.0",
        "stack": "LangGraph + LlamaIndex + Claude",
        "agent_ready": agent is not None,
        "documents_count": knowledge.get_document_count() if knowledge else 0,
    }


@app.post("/query", dependencies=[Depends(verify_api_key)])
def query_agent(request: QueryRequest):
    """
    Запрос к ИИ-юристу через LangGraph.

    Обычный def (не async): FastAPI выполнит его в threadpool,
    синхронные вызовы LLM/Neo4j/Chroma не заблокируют event loop.
    """
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Агент не инициализирован. Проверьте ANTHROPIC_API_KEY / OPENROUTER_API_KEY.",
        )

    try:
        return agent.process_query(request.question)
    except Exception:
        logger.exception("Ошибка обработки запроса")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/documents/search", dependencies=[Depends(verify_api_key)])
def search_documents(request: SearchRequest):
    """Поиск по базе знаний (только ретривал, без генерации)."""
    if not knowledge:
        raise HTTPException(status_code=503, detail="База знаний не инициализирована")

    try:
        results = knowledge.search(request.query, top_k=request.top_k)
        return {"results": results, "count": len(results)}
    except Exception:
        logger.exception("Ошибка поиска")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/documents/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    title: str = None,
    doc_type: str = None,
):
    """Загрузка документа в базу знаний через LlamaIndex."""
    if not knowledge or not pgs:
        raise HTTPException(status_code=503, detail="Система не инициализирована")

    # Проверка расширения
    suffix = f".{file.filename.rsplit('.', 1)[-1].lower()}" if "." in (file.filename or "") else ".txt"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Недопустимый тип файла '{suffix}'. Разрешены: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Проверка размера
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой. Лимит: {MAX_UPLOAD_MB} МБ",
        )

    import tempfile
    from src.pipeline.ingestion import IngestionPipeline

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = IngestionPipeline(knowledge=knowledge, pgs=pgs)
        # Тяжёлая синхронная работа — в threadpool, чтобы не блокировать event loop
        result = await run_in_threadpool(
            pipeline.ingest,
            tmp_path,
            metadata={"title": title or file.filename, "type": doc_type},
        )
        return result
    except Exception:
        logger.exception("Ошибка загрузки документа")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8080")),
        reload=True,
    )
