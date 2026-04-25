"""
Telegram-бот — первый интерфейс к ИИ-юристу.

Почему Telegram:
- Нулевой порог входа — ничего не нужно устанавливать
- Вирусность — ссылку на бота легко переслать
- Моментальная обратная связь от пользователей
- Поддержка файлов (можно отправить документ прямо в чат)
"""

import os
import asyncio
from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from loguru import logger
from dotenv import load_dotenv

from src.agent.core import LegalAgent
from src.pgs.graph import PGSGraph
from src.knowledge.store import KnowledgeStore

load_dotenv()

router = Router()

# Глобальные объекты (инициализируются при запуске)
agent: LegalAgent | None = None

# Храним контекст ожидания уточнений по user_id
pending_clarifications: dict[int, dict] = {}


# === КОМАНДЫ ===

@router.message(Command("start"))
async def cmd_start(message: Message):
    """Приветствие нового пользователя."""
    await message.answer(
        "\U0001f4da *ИИ-Юрист*\n\n"
        "Привет! Я — интеллектуальный юридический ассистент.\n\n"
        "Я могу помочь с:\n"
        "\u2022 Поиском нужных законов и статей\n"
        "\u2022 Составлением претензий и писем\n"
        "\u2022 Анализом договоров и правил\n"
        "\u2022 Разбором юридических ситуаций\n\n"
        "Просто напишите свой вопрос, и я постараюсь помочь.\n"
        "Если мне не хватит информации, я обязательно уточню \U0001f609",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Справка по командам."""
    await message.answer(
        "\U0001f4d6 *Команды*\n\n"
        "/start — начать сначала\n"
        "/help — эта справка\n"
        "/status — статус системы\n\n"
        "Также вы можете отправить документ (PDF, DOCX) — "
        "я проанализирую его и добавлю в базу знаний.",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("status"))
async def cmd_status(message: Message):
    """Статус системы."""
    kb_count = agent.knowledge.get_document_count() if agent else 0
    await message.answer(
        "\u2699\ufe0f *Статус системы*\n\n"
        f"Документов в базе знаний: {kb_count}\n"
        f"Агент: {'\u2705 активен (LangGraph + Claude)' if agent else '\u274c не инициализирован'}\n"
        f"RAG: {'\u2705 LlamaIndex' if agent else '\u26a0\ufe0f оффлайн'}\n"
        f"ПГС: {'\u2705 подключена' if agent and agent.pgs._driver else '\u26a0\ufe0f оффлайн'}",
        parse_mode=ParseMode.MARKDOWN,
    )


# === ОБРАБОТКА ТЕКСТОВЫХ СООБЩЕНИЙ ===

@router.message(F.text)
async def handle_text(message: Message):
    """Основной обработчик текстовых сообщений."""
    if not agent:
        await message.answer("\u26a0\ufe0f Система ещё не инициализирована. Попробуйте позже.")
        return

    user_id = message.from_user.id
    user_text = message.text.strip()

    # Показываем индикатор набора текста
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    logger.info(f"[Бот] Запрос от {user_id}: {user_text[:80]}...")

    try:
        # Запрос обрабатывается через LangGraph
        result = agent.process_query(user_text)

        if result["type"] == "clarification":
            # Почемучка: нужны уточнения
            pending_clarifications[user_id] = {
                "original_query": user_text,
                "questions": result["questions"],
            }
            questions_text = "\n".join(
                f"{i+1}. {q}" for i, q in enumerate(result["questions"])
            )
            await message.answer(
                f"\U0001f914 *Мне нужно уточнить:*\n\n{questions_text}\n\n"
                "Ответьте на эти вопросы, и я смогу дать точный ответ.",
                parse_mode=ParseMode.MARKDOWN,
            )

        elif result["type"] == "response":
            # Ответ готов
            confidence = result.get("confidence", 0)
            confidence_emoji = "\U0001f7e2" if confidence > 0.8 else "\U0001f7e1" if confidence > 0.5 else "\U0001f7e0"

            response_text = result["content"]

            # Добавляем информацию об источниках
            if result.get("sources"):
                response_text += f"\n\n—\n{confidence_emoji} Уверенность: {confidence:.0%}"
                response_text += f"\n\U0001f4c4 Источников: {len(result['sources'])}"

            # Telegram ограничивает сообщение 4096 символами
            if len(response_text) > 4000:
                parts = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                for part in parts:
                    await message.answer(part)
            else:
                await message.answer(response_text)

            # Очищаем ожидание уточнений
            pending_clarifications.pop(user_id, None)

    except Exception as e:
        logger.error(f"[Бот] Ошибка: {e}")
        await message.answer(
            "\u274c Произошла ошибка при обработке запроса. "
            "Попробуйте переформулировать вопрос."
        )


# === ОБРАБОТКА ДОКУМЕНТОВ ===

@router.message(F.document)
async def handle_document(message: Message):
    """Приём документов для загрузки в базу знаний."""
    doc = message.document
    file_name = doc.file_name or "unknown"
    file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    supported = {"pdf", "docx", "txt"}
    if file_ext not in supported:
        await message.answer(
            f"\u26a0\ufe0f Формат .{file_ext} пока не поддерживается.\n"
            f"Поддерживаемые форматы: {', '.join(supported)}"
        )
        return

    await message.answer(f"\U0001f4e5 Принял файл *{file_name}*. Обрабатываю...", parse_mode=ParseMode.MARKDOWN)

    try:
        # Скачиваем файл
        file = await message.bot.get_file(doc.file_id)
        file_bytes = await message.bot.download_file(file.file_path)

        # Сохраняем во временную директорию
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
            tmp.write(file_bytes.read())
            tmp_path = tmp.name

        # Загружаем через конвейер
        from src.pipeline.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(
            knowledge=agent.knowledge,
            pgs=agent.pgs,
        )
        result = pipeline.ingest(tmp_path, metadata={"title": file_name})

        # Удаляем временный файл
        os.unlink(tmp_path)

        if result["status"] == "ok":
            await message.answer(
                f"\u2705 Документ *{file_name}* успешно загружен!\n"
                f"Тип: {result['doc_type']}\n"
                f"ID: `{result['doc_id']}`",
                parse_mode=ParseMode.MARKDOWN,
            )
        elif result["status"] == "needs_clarification":
            questions_text = "\n".join(
                f"{i+1}. {q}" for i, q in enumerate(result["questions"])
            )
            await message.answer(
                f"\U0001f914 Я получил документ *{file_name}*, "
                f"но мне нужна помощь:\n\n"
                f"{questions_text}",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await message.answer(f"\u274c Ошибка: {result.get('message', 'неизвестная ошибка')}")

    except Exception as e:
        logger.error(f"[Бот] Ошибка загрузки документа: {e}")
        await message.answer("\u274c Не удалось обработать документ. Попробуйте ещё раз.")


# === ЗАПУСК ===

async def main():
    """Точка входа Telegram-бота."""
    global agent

    # Получаем токены из окружения
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not tg_token:
        logger.error("Не задан TELEGRAM_BOT_TOKEN")
        return
    if not anthropic_key:
        logger.error("Не задан ANTHROPIC_API_KEY")
        return

    # Инициализация компонентов
    neo4j_uri = os.getenv("NEO4J_URI")
    pgs = PGSGraph(
        uri=neo4j_uri,
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    ) if neo4j_uri else PGSGraph()

    chroma_host = os.getenv("CHROMA_HOST")
    knowledge = KnowledgeStore(
        host=chroma_host,
        port=int(os.getenv("CHROMA_PORT", "8000")),
    ) if chroma_host else KnowledgeStore()

    # Агент теперь на LangGraph
    agent = LegalAgent(
        api_key=anthropic_key,
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        pgs=pgs,
        knowledge=knowledge,
    )

    # Запуск бота
    bot = Bot(token=tg_token)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("\U0001f916 Telegram-бот ИИ-Юрист запущен (LangGraph + LlamaIndex)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
