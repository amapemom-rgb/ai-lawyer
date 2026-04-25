#!/usr/bin/env python3
"""
Автономная демка поиска по базе знаний ИИ-Юриста.
Работает без внешних зависимостей — использует TF-IDF для поиска.
"""

import os
import re
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "initial_docs"

STOP_WORDS = {
    "и", "в", "на", "с", "по", "к", "за", "из", "от", "до", "для", "о", "об",
    "не", "но", "а", "или", "что", "как", "это", "его", "её", "их", "ему",
    "ей", "им", "при", "все", "так", "же", "ли", "бы", "то", "он", "она",
    "оно", "они", "мы", "вы", "я", "ты", "быть", "был", "была", "были",
    "будет", "может", "если", "уже", "ещё", "еще", "также", "того",
    "этот", "эта", "эти", "тот", "та", "те", "этого", "свой", "свою",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r'[а-яёa-z0-9]+', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


class SimpleSearchEngine:
    def __init__(self):
        self.documents = []
        self.idf = {}

    def add_document(self, doc_id: str, title: str, content: str):
        tokens = tokenize(content)
        self.documents.append({"id": doc_id, "title": title, "content": content, "tokens": tokens, "tf": Counter(tokens)})

    def build_index(self):
        n = len(self.documents)
        all_terms = set()
        for doc in self.documents:
            all_terms.update(doc["tf"].keys())
        for term in all_terms:
            df = sum(1 for doc in self.documents if term in doc["tf"])
            self.idf[term] = math.log((n + 1) / (df + 1)) + 1

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_tf = Counter(query_tokens)
        results = []
        for doc in self.documents:
            score = 0
            for term, qtf in query_tf.items():
                if term in doc["tf"]:
                    tf = doc["tf"][term] / len(doc["tokens"]) if doc["tokens"] else 0
                    idf = self.idf.get(term, 1)
                    score += qtf * tf * idf
            if score > 0:
                snippet = self._find_snippet(doc["content"], query_tokens)
                results.append({"title": doc["title"], "score": round(score, 4), "snippet": snippet})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _find_snippet(self, content, query_tokens, window=300):
        content_lower = content.lower()
        best_pos, best_count = 0, 0
        for i in range(0, len(content_lower), 100):
            chunk = content_lower[i:i + window]
            count = sum(1 for t in query_tokens if t in chunk)
            if count > best_count:
                best_count = count
                best_pos = i
        snippet = content[best_pos:best_pos + window].strip()
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + window < len(content):
            snippet = snippet + "..."
        return snippet


def main():
    print()
    print("=" * 60)
    print("  ⚖️  ИИ-ЮРИСТ — демо поиска по базе знаний")
    print("  (автономный режим, без внешних зависимостей)")
    print("=" * 60)
    print()

    engine = SimpleSearchEngine()
    doc_names = {
        "01_zozpp_key_articles.txt": "Закон о защите прав потребителей",
        "02_gk_rf_sale_contract.txt": "ГК РФ — купля-продажа",
        "03_distance_selling_rules.txt": "Правила дистанционной продажи",
        "04_pretension_template.txt": "Шаблон претензии",
        "05_marketplace_liability.txt": "Ответственность маркетплейсов",
        "06_wb_ozon_common_rules.txt": "Правила WB, Ozon, Яндекс Маркет",
        "07_ecommerce_faq.txt": "FAQ по e-commerce",
        "08_pretension_sending_guide.txt": "Как отправить претензию",
    }

    print("📚 Загрузка базы знаний...")
    for filename, title in doc_names.items():
        filepath = DOCS_DIR / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            engine.add_document(filename, title, content)
            print(f"   ✅ {title}")
        else:
            print(f"   ❌ Не найден: {filename}")

    engine.build_index()
    print(f"\n   📊 Загружено: {len(engine.documents)} документов")
    print()

    test_queries = [
        "Как вернуть товар на Wildberries?",
        "Куда отправить претензию на Ozon?",
        "Какая неустойка за просрочку возврата денег?",
        "Можно ли вернуть товар без чека?",
        "Кто отвечает за качество товара — маркетплейс или продавец?",
    ]

    print("-" * 60)
    print("🔍 Тестовые запросы:\n")
    for query in test_queries:
        print(f"❓ {query}")
        results = engine.search(query, top_k=3)
        if not results:
            print("   🤷 Ничего не найдено\n")
            continue
        for i, r in enumerate(results, 1):
            print(f"   {i}. [{r['title']}] (score: {r['score']})")
            print(f"      {r['snippet'][:150]}...")
        print()

    print("-" * 60)
    print("Теперь ваша очередь! Задавайте вопросы.")
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
        results = engine.search(query, top_k=5)
        if not results:
            print("   🤷 Ничего не найдено.")
            continue
        print(f"\n   📄 Найдено {len(results)} фрагментов:\n")
        for i, r in enumerate(results, 1):
            print(f"   {i}. [{r['title']}] (score: {r['score']})")
            print(f"      {r['snippet'][:200]}...")
            print()


if __name__ == "__main__":
    main()
