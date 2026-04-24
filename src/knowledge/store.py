"""
База знаний — векторное хранилище на ChromaDB.

Хранит документы (законы, договоры, правила, переписку)
и обеспечивает семантический поиск по ним.
"""

import chromadb
from loguru import logger


class KnowledgeStore:
    """Векторное хранилище документов."""

    def __init__(self, host: str = None, port: int = 8000, collection_name: str = "legal_knowledge"):
        if host:
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            self.client = chromadb.Client()  # in-memory для разработки

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"База знаний: коллекция '{collection_name}' | документов: {self.collection.count()}")

    def add_document(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        doc_type: str = "",
        metadata: dict = None,
    ):
        """Добавление документа в базу знаний."""
        meta = metadata or {}
        meta.update({"title": title, "type": doc_type})

        chunks = self._split_into_chunks(content)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**meta, "chunk_index": i, "parent_id": doc_id} for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(f"Добавлен документ '{title}' | {len(chunks)} чанков")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Семантический поиск по базе знаний."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        documents = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                documents.append({
                    "id": results["ids"][0][i] if results["ids"] else "",
                    "content": doc,
                    "title": meta.get("title", ""),
                    "type": meta.get("type", ""),
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })

        return documents

    def update_document(self, doc_id: str, content: str, metadata: dict = None):
        """Обновление существующего документа."""
        existing = self.collection.get(where={"parent_id": doc_id})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        title = (metadata or {}).get("title", "")
        doc_type = (metadata or {}).get("type", "")
        self.add_document(doc_id, content, title, doc_type, metadata)
        logger.info(f"Документ '{doc_id}' обновлён")

    def _split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Разбиение текста на перекрывающиеся чанки."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks
