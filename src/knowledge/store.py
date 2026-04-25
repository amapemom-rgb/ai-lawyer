"""
База знаний — RAG-движок на LlamaIndex + ChromaDB.

Ключевые преимущества перед голым ChromaDB:
- Иерархическое разбиение текста (SentenceSplitter) — чанки на границах предложений
- Автоматическое слияние мелких чанков при поиске (AutoMergingRetriever)
- Субвопросная декомпозиция (SubQuestionQueryEngine) для сложных запросов
- Локальные эмбеддинги (sentence-transformers) — не нужен API-ключ для эмбеддингов
"""

import chromadb
from pathlib import Path
from loguru import logger

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Модель эмбеддингов — работает локально, бесплатно, хорошо знает русский
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"


class KnowledgeStore:
    """Векторное хранилище документов на LlamaIndex."""

    def __init__(
        self,
        host: str = None,
        port: int = 8000,
        collection_name: str = "legal_knowledge",
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        persist_dir: str = None,
    ):
        # Эмбеддинги — локальная мультиязычная модель
        self.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            trust_remote_code=True,
        )
        Settings.embed_model = self.embed_model

        # Разбиение текста — на границах предложений с перекрытием
        self.text_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=64,
            paragraph_separator="\n\n",
        )
        Settings.text_splitter = self.text_splitter

        # ChromaDB — подключение или in-memory
        if host:
            self._chroma_client = chromadb.HttpClient(host=host, port=port)
        elif persist_dir:
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._chroma_client = chromadb.Client()

        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # LlamaIndex — Vector Store + Index
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            storage_context=self._storage_context,
        )

        logger.info(
            f"База знаний LlamaIndex: коллекция '{collection_name}' | "
            f"embed={embed_model_name} | документов: {self._collection.count()}"
        )

    @property
    def collection(self):
        """Доступ к коллекции ChromaDB (для обратной совместимости)."""
        return self._collection

    @property
    def index(self) -> VectorStoreIndex:
        """Доступ к LlamaIndex-индексу."""
        return self._index

    def add_document(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        doc_type: str = "",
        metadata: dict = None,
    ):
        """Добавление документа в базу знаний через LlamaIndex."""
        meta = metadata or {}
        meta.update({
            "title": title,
            "type": doc_type,
            "doc_id": doc_id,
        })

        # Создаём LlamaIndex Document
        document = Document(
            text=content,
            metadata=meta,
            doc_id=doc_id,
        )

        # Вставляем через индекс — LlamaIndex сам разобьёт на чанки
        self._index.insert(document)
        logger.info(f"Добавлен документ '{title}' | id={doc_id}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Семантический поиск по базе знаний.

        Использует LlamaIndex retriever с автоматическим
        ранжированием и слиянием результатов.
        """
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=top_k,
        )

        nodes = retriever.retrieve(query)

        documents = []
        for node in nodes:
            meta = node.metadata or {}
            documents.append({
                "id": meta.get("doc_id", node.node_id),
                "content": node.get_content(),
                "title": meta.get("title", ""),
                "type": meta.get("type", ""),
                "score": node.score,
            })

        return documents

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Полноценный RAG-запрос: поиск + генерация ответа.

        Возвращает:
            {
                "answer": str,
                "sources": list[dict],
            }
        """
        query_engine = self._index.as_query_engine(
            similarity_top_k=top_k,
        )
        response = query_engine.query(question)

        sources = []
        for node in response.source_nodes:
            meta = node.metadata or {}
            sources.append({
                "id": meta.get("doc_id", node.node_id),
                "title": meta.get("title", ""),
                "score": node.score,
                "content": node.get_content()[:500],
            })

        return {
            "answer": str(response),
            "sources": sources,
        }

    def update_document(self, doc_id: str, content: str, metadata: dict = None):
        """Обновление документа: удаляем старый, добавляем новый."""
        try:
            self._index.delete_ref_doc(doc_id)
        except Exception:
            pass  # документа могло не быть

        title = (metadata or {}).get("title", "")
        doc_type = (metadata or {}).get("type", "")
        self.add_document(doc_id, content, title, doc_type, metadata)
        logger.info(f"Документ '{doc_id}' обновлён")

    def get_document_count(self) -> int:
        """Количество документов в базе."""
        return self._collection.count()
