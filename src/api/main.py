"""
FastAPI — точка входа.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Lawyer запускается...")
    # TODO: инициализация подключений к Neo4j, PostgreSQL, ChromaDB
    yield
    logger.info("AI Lawyer остановлен")


app = FastAPI(
    title="AI Lawyer — ИИ-Юрист",
    description="Интеллектуальная юридическая система с понятийно-графовой архитектурой",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/query")
async def query(request: dict):
    """
    Основной эндпоинт для запросов к ИИ-юристу.
    TODO: подключить LegalAgent
    """
    return {"message": "Endpoint ready, agent not connected yet"}


@app.post("/documents/upload")
async def upload_document():
    """
    Загрузка документа в базу знаний.
    TODO: подключить IngestionPipeline
    """
    return {"message": "Upload endpoint ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8080")),
        reload=True,
    )
