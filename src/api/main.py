"""
FastAPI — REST API для ИИ-юриста.

Эндпоинты:
- GET  /health          — проверка состояния
- POST /query           — запрос к ИИ-юристу (LangGraph)
- POST /documents/upload — загрузка документа в базу знаний
- GET  /documents/search — поиск по базе знаний
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

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

    # База знаний (LlamaIndex + ChromaDB)
    chroma_host = os.getenv("CHROMA_HOST")
    knowledge = KnowledgeStore(
        host=chroma_host,
        port=int(os.getenv("CHROMA_PORT", "8000")),
    ) if chroma_host else KnowledgeStore()

    # Агент (LangGraph + Claude)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        agent = LegalAgent(
            api_key=anthropic_key,
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            pgs=pgs,
            knowledge=knowledge,
        )
        logger.info("Агент LangGraph инициализирован")
    else:
        logger.warning("ANTHROPIC_API_KEY не задан — агент не запущен")

    yield

    if pgs:
        pgs.close()
    logger.info("AI Lawyer остановлен")


app = FastAPI(
    title="AI Lawyer — ИИ-Юрист",
    description="Юридическая система: LangGraph (оркестрация) + LlamaIndex (RAG) + Claude (генерация)",
    version="0.2.0",
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
        "version": "0.2.0",
        "stack": "LangGraph + LlamaIndex + Claude",
        "agent_ready": agent is not None,
        "documents_count": knowledge.get_document_count() if knowledge else 0,
    }


@app.post("/query")
async def query_agent(request: QueryRequest):
    """Запрос к ИИ-юристу через LangGraph."""
    if not agent:
        raise HTTPException(status_code=503, detail="Агент не инициализирован. Проверьте ANTHROPIC_API_KEY.")

    try:
        result = agent.process_query(request.question)
        return result
    except Exception as e:
        logger.error(f"Ошибка запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/search")
async def search_documents(request: SearchRequest):
    """Поиск по базе знаний (только ретривал, без генерации)."""
    if not knowledge:
        raise HTTPException(status_code=503, detail="База знаний не инициализирована")

    results = knowledge.search(request.query, top_k=request.top_k)
    return {"results": results, "count": len(results)}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = None,
    doc_type: str = None,
):
    """Загрузка документа в базу знаний через LlamaIndex."""
    if not knowledge or not pgs:
        raise HTTPException(status_code=503, detail="Система не инициализирована")

    import tempfile
    from src.pipeline.ingestion import IngestionPipeline

    # Сохраняем во временный файл
    suffix = f".{file.filename.rsplit('.', 1)[-1]}" if "." in file.filename else ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = IngestionPipeline(knowledge=knowledge, pgs=pgs)
        result = pipeline.ingest(
            tmp_path,
            metadata={"title": title or file.filename, "type": doc_type},
        )
        return result
    except Exception as e:
        logger.error(f"Ошибка загрузки документа: {e}")
        raise HTTPException(status_code=500, detail=str(e))
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
