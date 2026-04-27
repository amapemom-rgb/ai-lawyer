#!/usr/bin/env python3
"""
Облегчённая демка ИИ-Юриста: TF-IDF поиск + OpenRouter (Claude).
Не требует тяжёлых моделей — работает на любом компьютере.

Запуск:
    pip3 install openai python-dotenv --break-system-packages
    python3 scripts/demo_openrouter.py

Нужен файл .env с ключом:
    OPENROUTER_API_KEY=sk-or-...
"""

import os
import re
import math
from pathlib import Path
from collections import Counter

# === Конфигурация ===

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "initial_docs"

# Загружаем .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # python-dotenv не обязателен, можно задать переменные вручную


# === Стоп-слова ===

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


# === TF-IDF поиск ===

class SimpleSearchEngine:
    def __init__(self):
        self.documents = []
        self.idf = {}

    def add_document(self, doc_id: str, title: str, content: str):
        tokens = tokenize(content)
        self.documents.append({
            "id": doc_id,
            "title": title,
            "content": content,
            "tokens": tokens,
            "tf": Counter(tokens),
        })

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
                results.append({
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": round(score, 4),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# === OpenRouter (Claude) ===

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


def ask_llm(query: str, context: str, api_key: str, model: str) -> str:
    """Отправляет запрос в OpenRouter и возвращает ответ."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/amapemom-rgb/ai-lawyer",
            "X-Title": "AI-Lawyer",
        },
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if context:
        messages.append({
            "role": "user",
            "content": f"Контекст из базы знаний:\n{context}",
        })

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
    )

    return response.choices[0].message.content


# === Загрузка документов ===

def load_documents(engine: SimpleSearchEngine):
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

    for filename, title in doc_names.items():
        filepath = DOCS_DIR / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            engine.add_document(filename, title, content)
            print(f"   ✅ {title}")
        else:
            print(f"   ❌ Не найден: {filename}")

    engine.build_index()


# === Главная функция ===

def main():
    print()
    print("=" * 60)
    print("  ⚖️  ИИ-ЮРИСТ  —  полный режим (поиск + ответ)")
    print("  Стек: TF-IDF (поиск) + OpenRouter/Claude (ответ)")
    print("=" * 60)
    print()

    # Проверяем ключ
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY не найден!")
        print("   Создайте файл .env в корне проекта:")
        print("   OPENROUTER_API_KEY=sk-or-ваш-ключ")
        print()
        print("   Получить ключ: https://openrouter.ai/keys")
        return

    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")
    print(f"🤖 Модель: {model}")
    print()

    # Загрузка документов
    print("📚 Загрузка базы знаний...")
    engine = SimpleSearchEngine()
    load_documents(engine)
    print(f"\n   📊 Загружено: {len(engine.documents)} документов")
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

        # Шаг 1: Поиск
        results = engine.search(query, top_k=5)

        if results:
            # Формируем контекст из найденных документов
            context_parts = []
            print("\n   📄 Найденные источники:")
            for i, r in enumerate(results, 1):
                print(f"      {i}. {r['title']} (релевантность: {r['score']})")
                # Берём первые 1500 символов каждого документа
                context_parts.append(f"--- {r['title']} ---\n{r['content'][:1500]}")
            context = "\n\n".join(context_parts)
        else:
            context = ""
            print("\n   📄 В базе знаний ничего не найдено по этому запросу.")

        # Шаг 2: Генерация ответа через Claude
        print("\n   ⏳ Генерирую ответ...")
        try:
            answer = ask_llm(query, context, api_key, model)
            print(f"\n   💬 Ответ:\n")
            # Красиво выводим с отступом
            for line in answer.split("\n"):
                print(f"   {line}")
        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg or "insufficient" in error_msg.lower():
                print(f"\n   ❌ Недостаточно средств на OpenRouter. Пополните баланс.")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                print(f"\n   ❌ Неверный API-ключ. Проверьте OPENROUTER_API_KEY в .env")
            else:
                print(f"\n   ❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
