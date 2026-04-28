#!/usr/bin/env python3
"""
ИИ-Юрист: семантический поиск + OpenRouter (Claude).
Фаза 1.5 — улучшенный поиск через эмбеддинги, без тяжёлых локальных моделей.

Запуск:
    pip3 install openai python-dotenv numpy --break-system-packages
    python3 scripts/demo_openrouter.py

Нужен файл .env с ключом:
    OPENROUTER_API_KEY=sk-or-...

Как работает:
    1. При первом запуске документы разбиваются на чанки и отправляются
       на OpenRouter Embeddings API (text-embedding-3-small, ~$0.02/1M токенов).
       Векторы кэшируются в data/embeddings_cache.json.
    2. При каждом вопросе: запрос → эмбеддинг → cosine similarity → топ-5 чанков.
    3. Найденные чанки отправляются как контекст в Claude для генерации ответа.
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path

# === Конфигурация ===

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "initial_docs"
CACHE_FILE = ROOT / "data" / "embeddings_cache.json"

EMBED_MODEL = "openai/text-embedding-3-small"
CHUNK_SIZE = 500  # символов
CHUNK_OVERLAP = 100  # символов перехлёста

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# === Системный промпт ===

SYSTEM_PROMPT = """Ты — ИИ-юрист, специализирующийся на российском праве в сфере e-commerce.

ПРАВИЛА:
1. Отвечай точно, ссылайся на конкретные статьи законов и документы из контекста.
2. Если информации в контексте недостаточно — скажи об этом прямо.
3. НИКОГДА не выдумывай номера статей, законов или нормативных актов.
4. Если вопрос выходит за рамки предоставленного контекста — так и скажи.
5. Используй простой, понятный язык. Юридические термины объясняй.
6. Структурируй ответ: сначала прямой ответ, потом обоснование со ссылками.

ВАЖНЫЕ ФАКТЫ ДЛЯ ВАЛИДАЦИИ:
- В Законе о защите прав потребителей (ЗоЗПП) всего 46 статей. Если спрашивают о статье с номером больше 46 — такой статьи НЕ СУЩЕСТВУЕТ, скажи об этом прямо.
- Статья 26.1 ЗоЗПП — дистанционная торговля, срок возврата 7 дней.
- Статья 18 ЗоЗПП — права при обнаружении недостатков, отсутствие чека не основание для отказа.
- Статья 22 ЗоЗПП — срок возврата денег 10 дней.
- Статья 23 ЗоЗПП — неустойка 1% в день.

ФОРМАТ ОТВЕТА:
- Прямой ответ на вопрос (1-2 предложения)
- Обоснование со ссылками на конкретные статьи/документы
- Практический совет (что делать дальше)
"""


# === Разбивка на чанки ===

def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Разбивает текст на перекрывающиеся чанки."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Стараемся резать по границе предложения
        if end < len(text):
            # Ищем последнюю точку/перенос строки в пределах чанка
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.3:  # не слишком рано
                    end = start + last_sep + len(sep)
                    break

        chunk_text_content = text[start:end].strip()
        if len(chunk_text_content) > 50:  # пропускаем слишком короткие
            chunks.append({
                "title": title,
                "content": chunk_text_content,
            })

        start = end - overlap if end < len(text) else len(text)

    return chunks


# === Эмбеддинги через OpenRouter ===

def get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """Получает эмбеддинги через OpenRouter API."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Батчим по 50 текстов (лимит API)
    all_embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# === Семантический поисковый движок ===

class SemanticSearchEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chunks = []       # [{title, content}]
        self.embeddings = None  # numpy array, shape (n_chunks, embed_dim)
        self._cache_hash = None

    def load_documents(self):
        """Загрузка и разбивка документов на чанки."""
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

        all_content = ""
        for filename, title in doc_names.items():
            filepath = DOCS_DIR / filename
            if filepath.exists():
                content = filepath.read_text(encoding="utf-8")
                all_content += content
                doc_chunks = chunk_text(content, title)
                self.chunks.extend(doc_chunks)
                print(f"   ✅ {title} ({len(doc_chunks)} чанков)")
            else:
                print(f"   ❌ Не найден: {filename}")

        # Хеш для проверки — изменились ли документы
        self._cache_hash = hashlib.md5(all_content.encode()).hexdigest()

    def build_index(self):
        """Создание или загрузка эмбеддингов."""
        # Проверяем кэш
        if CACHE_FILE.exists():
            try:
                cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if cache.get("hash") == self._cache_hash:
                    self.embeddings = np.array(cache["embeddings"])
                    print(f"\n   📦 Загружен кэш эмбеддингов ({len(self.chunks)} чанков)")
                    return
            except Exception:
                pass  # кэш повреждён — пересоздадим

        # Создаём эмбеддинги
        print(f"\n   🔄 Создаю эмбеддинги для {len(self.chunks)} чанков...")
        print(f"      (модель: {EMBED_MODEL}, это займёт несколько секунд)")

        texts = [chunk["content"] for chunk in self.chunks]
        embeddings = get_embeddings(texts, self.api_key)
        self.embeddings = np.array(embeddings)

        # Сохраняем кэш
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "hash": self._cache_hash,
            "embeddings": [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings],
        }
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
        print(f"      ✅ Кэш сохранён в {CACHE_FILE.name}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Семантический поиск по cosine similarity."""
        # Получаем эмбеддинг запроса
        query_embedding = np.array(get_embeddings([query], self.api_key)[0])

        # Считаем сходство со всеми чанками
        scores = []
        for i in range(len(self.chunks)):
            score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((i, score))

        # Сортируем по убыванию и берём top_k
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        title_count = {}
        for idx, score in scores[:top_k * 3]:  # берём больше, чтобы разнообразить
            chunk = self.chunks[idx]
            title = chunk["title"]

            # Не больше 2 чанков из одного документа
            title_count[title] = title_count.get(title, 0) + 1
            if title_count[title] > 2:
                continue

            results.append({
                "title": title,
                "content": chunk["content"],
                "score": round(score, 4),
            })

            if len(results) >= top_k:
                break

        return results


# === Вызов LLM ===

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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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


# === Главная функция ===

def main():
    print()
    print("=" * 60)
    print("  ⚖️  ИИ-ЮРИСТ  —  Фаза 1.5 (семантический поиск)")
    print("  Стек: OpenRouter Embeddings + Claude")
    print("=" * 60)
    print()

    # Проверяем ключ
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY не найден!")
        print("   Создайте файл .env: OPENROUTER_API_KEY=sk-or-ваш-ключ")
        return

    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
    print(f"🤖 Модель ответов: {model}")
    print(f"🔍 Модель поиска:  {EMBED_MODEL}")
    print()

    # Загрузка и индексация документов
    print("📚 Загрузка базы знаний...")
    engine = SemanticSearchEngine(api_key)
    engine.load_documents()
    engine.build_index()
    print(f"\n   📊 Итого: {len(engine.chunks)} чанков из {8} документов")
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

        # Шаг 1: Семантический поиск
        results = engine.search(query, top_k=5)

        if results:
            context_parts = []
            print("\n   📄 Найденные источники:")
            for i, r in enumerate(results, 1):
                print(f"      {i}. {r['title']} (сходство: {r['score']})")
                context_parts.append(f"--- {r['title']} ---\n{r['content']}")
            context = "\n\n".join(context_parts)
        else:
            context = ""
            print("\n   📄 В базе знаний ничего не найдено.")

        # Шаг 2: Генерация ответа
        print("\n   ⏳ Генерирую ответ...")
        try:
            answer = ask_llm(query, context, api_key, model)
            print(f"\n   💬 Ответ:\n")
            for line in answer.split("\n"):
                print(f"   {line}")
        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg:
                print(f"\n   ❌ Недостаточно средств на OpenRouter.")
            elif "401" in error_msg:
                print(f"\n   ❌ Неверный API-ключ.")
            else:
                print(f"\n   ❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
