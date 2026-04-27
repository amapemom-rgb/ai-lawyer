#!/usr/bin/env python3
"""
Быстрая демка ИИ-Юриста в терминале.

Запуск:
    pip3 install -r requirements.txt
    python3 scripts/demo_cli.py

Без API-ключа:  покажет только результаты поиска по базе знаний
С API-ключом:   полный RAG — поиск + ответ LLM со ссылками

Поддерживаемые провайдеры (в файле .env):
    OPENROUTER_API_KEY=sk-or-...   (OpenRouter — дешевле, много моделей)
    ANTHROPIC_API_KEY=sk-ant-...    (Anthropic напрямую)

Опционально:
    OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514  (модель для OpenRouter)
    CLAUDE_MODEL=claude-sonnet-4-20250514                (модель для Anthropic)
"""

import os
import sys
from pathlib import Path

# Корень проекта
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def print_banner():
    print()
    print("=" * 60)
    print("  ⚖️  ИИ-ЮРИСТ  —  демо в терминале")
    print("  Стек: LlamaIndex (RAG) + LLM (генерация)")
    print("=" * 60)
    print()


def load_knowledge():
    """Загрузка базы знаний в память."""
    from src.knowledge.store import KnowledgeStore

    print("📚 Загружаю базу знаний...")
    print("   (первый запуск скачает модель эмбеддингов ~1 ГБ)")
    print()

    store = KnowledgeStore()

    docs_dir = ROOT / "data" / "initial_docs"
    documents = [
        ("zozpp_key_articles", "01_zozpp_key_articles.txt", "Закон о защите прав потребителей", "law"),
        ("gk_rf_sale", "02_gk_rf_sale_contract.txt", "ГК РФ — купля-продажа", "law"),
        ("distance_selling", "03_distance_selling_rules.txt", "Правила дистанционной продажи", "regulation"),
        ("pretension_template", "04_pretension_template.txt", "Шаблон претензии", "instruction"),
        ("marketplace_liability", "05_marketplace_liability.txt", "Ответственность маркетплейсов", "regulation"),
        ("wb_ozon_rules", "06_wb_ozon_common_rules.txt", "Правила WB, Ozon, Яндекс Маркет", "regulation"),
        ("ecommerce_faq", "07_ecommerce_faq.txt", "FAQ по e-commerce", "instruction"),
        ("pretension_sending", "08_pretension_sending_guide.txt", "Как отправить претензию", "instruction"),
    ]

    for doc_id, filename, title, doc_type in documents:
        filepath = docs_dir / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            store.add_document(doc_id=doc_id, content=content, title=title, doc_type=doc_type)
            print(f"   ✅ {title}")
        else:
            print(f"   ❌ Не найден: {filename}")

    print(f"\n   📊 Загружено записей: {store.get_document_count()}")
    return store


def detect_provider():
    """Определяет доступный LLM-провайдер."""
    if os.getenv("OPENROUTER_API_KEY"):
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")
        return "openrouter", model
    elif os.getenv("ANTHROPIC_API_KEY"):
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        return "anthropic", model
    else:
        return None, None


def run_search_only(store):
    """Режим без API-ключа: только поиск по базе знаний."""
    print()
    print("⚠️  API-ключ не найден")
    print("   Работаю в режиме поиска (без генерации ответов)")
    print()
    print("   Чтобы получить полные ответы, добавьте в .env один из ключей:")
    print("   OPENROUTER_API_KEY=sk-or-...  (OpenRouter, от $5)")
    print("   ANTHROPIC_API_KEY=sk-ant-...  (Anthropic напрямую)")
    print()
    print("   OpenRouter: https://openrouter.ai/keys")
    print("   Anthropic:  https://console.anthropic.com/settings/keys")
    print()
    print("-" * 60)
    print("Задавайте вопросы. Я покажу, что нашёл в базе знаний.")
    print("Для выхода: quit / exit / выход")
    print("-" * 60)

    while True:
        print()
        try:
            query = input("❓ Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 До встречи!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "выход", "q"):
            print("👋 До встречи!")
            break

        results = store.search(query, top_k=5)

        if not results:
            print("   🤷 Ничего не найдено. Попробуйте другой вопрос.")
            continue

        print(f"\n   📄 Найдено {len(results)} релевантных фрагментов:\n")
        for i, doc in enumerate(results, 1):
            title = doc.get("title", "—")
            score = doc.get("score")
            score_str = f" (релевантность: {score:.3f})" if score is not None else ""
            content = doc.get("content", "")[:300].replace("\n", " ")
            print(f"   {i}. [{title}]{score_str}")
            print(f"      {content}...")
            print()


def run_full_agent(store, provider, model):
    """Режим с API-ключом: полный RAG через LangGraph."""
    from src.agent.core import LegalAgent
    from src.pgs.graph import PGSGraph

    provider_name = "OpenRouter" if provider == "openrouter" else "Anthropic"
    print(f"\n🤖 Инициализация агента ({provider_name}, модель: {model})...")

    pgs = PGSGraph()

    # create_llm() сам определит провайдера по переменным окружения
    agent = LegalAgent(
        pgs=pgs,
        knowledge=store,
    )

    print(f"   ✅ Агент готов (LangGraph + {provider_name})")
    print()
    print("-" * 60)
    print("Задавайте юридические вопросы по e-commerce.")
    print("Я найду нужные законы и дам ответ со ссылками.")
    print("Для выхода: quit / exit / выход")
    print("-" * 60)

    while True:
        print()
        try:
            query = input("❓ Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 До встречи!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "выход", "q"):
            print("👋 До встречи!")
            break

        print("\n   ⏳ Думаю...")

        try:
            result = agent.process_query(query)

            if result.get("type") == "clarification":
                print("\n   🤔 Мне нужно уточнить:")
                for i, q in enumerate(result.get("questions", []), 1):
                    print(f"      {i}. {q}")

            elif result.get("type") == "response":
                confidence = result.get("confidence", 0)
                emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🟠"

                print(f"\n   {emoji} Ответ (уверенность: {confidence:.0%}):\n")
                print("   " + result.get("content", "").replace("\n", "\n   "))

                sources = result.get("sources", [])
                if sources:
                    print(f"\n   📄 Источников использовано: {len(sources)}")
            else:
                print(f"\n   ⚠️ Неожиданный результат: {result}")

        except Exception as e:
            print(f"\n   ❌ Ошибка: {e}")
            print("   Попробуйте переформулировать вопрос.")


def main():
    print_banner()

    # Загрузка базы знаний
    store = load_knowledge()

    # Определяем провайдера
    provider, model = detect_provider()

    if provider:
        run_full_agent(store, provider, model)
    else:
        run_search_only(store)


if __name__ == "__main__":
    main()
