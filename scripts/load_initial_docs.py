"""
Скрипт загрузки начальных документов в базу знаний.

Запуск: python scripts/load_initial_docs.py

Загружает все файлы из data/initial_docs/ через LlamaIndex в ChromaDB.
Не требует Docker — использует in-memory хранилище или
локальный ChromaDB сервер, если указан в .env.
"""

import os
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger
from src.knowledge.store import KnowledgeStore

load_dotenv()

# Конфигурация документов
DOCUMENTS = [
    {
        "id": "zozpp_key_articles",
        "file": "01_zozpp_key_articles.txt",
        "title": "Закон о защите прав потребителей — ключевые статьи",
        "type": "law",
    },
    {
        "id": "gk_rf_sale",
        "file": "02_gk_rf_sale_contract.txt",
        "title": "ГК РФ — купля-продажа",
        "type": "law",
    },
    {
        "id": "distance_selling",
        "file": "03_distance_selling_rules.txt",
        "title": "Правила дистанционной продажи товаров",
        "type": "regulation",
    },
    {
        "id": "pretension_template",
        "file": "04_pretension_template.txt",
        "title": "Шаблон претензии к маркетплейсу",
        "type": "instruction",
    },
    {
        "id": "marketplace_liability",
        "file": "05_marketplace_liability.txt",
        "title": "Ответственность маркетплейсов по законодательству РФ",
        "type": "regulation",
    },
    {
        "id": "wb_ozon_rules",
        "file": "06_wb_ozon_common_rules.txt",
        "title": "Правила Wildberries, Ozon, Яндекс Маркет",
        "type": "regulation",
    },
    {
        "id": "ecommerce_faq",
        "file": "07_ecommerce_faq.txt",
        "title": "Частые юридические вопросы по e-commerce",
        "type": "instruction",
    },
    {
        "id": "pretension_sending",
        "file": "08_pretension_sending_guide.txt",
        "title": "Как правильно отправить претензию",
        "type": "instruction",
    },
]


def main():
    logger.info("\U0001f4da Загрузка начальной базы знаний (LlamaIndex)...")

    # Инициализация хранилища
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        knowledge = KnowledgeStore(
            host=chroma_host,
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
        logger.info(f"Подключено к ChromaDB серверу: {chroma_host}")
    else:
        knowledge = KnowledgeStore()
        logger.info("Используется in-memory хранилище (ChromaDB + LlamaIndex)")

    # Загрузка документов
    docs_dir = project_root / "data" / "initial_docs"
    loaded = 0
    errors = 0

    for doc_info in DOCUMENTS:
        file_path = docs_dir / doc_info["file"]

        if not file_path.exists():
            logger.error(f"\u274c Файл не найден: {file_path}")
            errors += 1
            continue

        content = file_path.read_text(encoding="utf-8")

        knowledge.add_document(
            doc_id=doc_info["id"],
            content=content,
            title=doc_info["title"],
            doc_type=doc_info["type"],
        )
        loaded += 1
        logger.info(f"\u2705 {doc_info['title']}")

    logger.info(f"\n\U0001f389 Загружено: {loaded} | Ошибок: {errors}")
    logger.info(f"\U0001f4ca Всего записей в базе: {knowledge.get_document_count()}")

    # Тестовый запрос через LlamaIndex retriever
    logger.info("\n\U0001f50d Тестовый запрос: 'возврат товара на маркетплейсе'")
    results = knowledge.search("возврат товара на маркетплейсе", top_k=3)

    for i, doc in enumerate(results, 1):
        title = doc.get("title", "без названия")
        score = doc.get("score", "?")
        preview = doc.get("content", "")[:100]
        logger.info(f"  {i}. [{title}] (score: {score})")
        logger.info(f"     {preview}...")

    logger.info("\n\u2705 База знаний готова (LlamaIndex + ChromaDB)!")


if __name__ == "__main__":
    main()
