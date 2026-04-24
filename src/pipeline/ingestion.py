"""
Входной конвейер документов.

Отвечает за:
1. Парсинг документов (PDF, DOCX, TXT)
2. Классификацию (новый документ / изменение / дополнение)
3. Загрузку в базу знаний и ПГС
4. Вызов «Почемучки» если документ не удаётся классифицировать
"""

from pathlib import Path
from loguru import logger
from src.knowledge.store import KnowledgeStore
from src.pgs.graph import PGSGraph


class DocumentType:
    LAW = "law"
    CONTRACT = "contract"
    REGULATION = "regulation"
    INSTRUCTION = "instruction"
    CORRESPONDENCE = "correspondence"
    AMENDMENT = "amendment"
    UNKNOWN = "unknown"


class IngestionPipeline:
    """Конвейер загрузки и обработки документов."""

    def __init__(self, knowledge: KnowledgeStore, pgs: PGSGraph):
        self.knowledge = knowledge
        self.pgs = pgs

    def ingest(self, file_path: str, metadata: dict = None) -> dict:
        """
        Полный цикл обработки документа.

        Returns:
            {
                "status": "ok" | "needs_clarification",
                "doc_id": str,
                "doc_type": str,
                "questions": list[str] | None,
            }
        """
        path = Path(file_path)
        logger.info(f"Загрузка документа: {path.name}")

        # 1. Парсинг
        content = self._parse(path)
        if not content:
            return {"status": "error", "message": f"Не удалось распарсить {path.name}"}

        # 2. Классификация
        doc_type = self._classify(content, metadata)

        # 3. Если не удалось классифицировать — спрашиваем
        if doc_type == DocumentType.UNKNOWN:
            return {
                "status": "needs_clarification",
                "doc_id": None,
                "doc_type": doc_type,
                "questions": [
                    "Что это за документ? (закон, договор, правило, инструкция, переписка)",
                    "К какому существующему документу это относится?",
                    "Это новый документ или изменение к существующему?",
                ],
            }

        # 4. Генерация ID
        doc_id = self._generate_id(path.name, doc_type)

        # 5. Проверяем: это изменение к существующему документу?
        if doc_type == DocumentType.AMENDMENT and metadata:
            original_id = metadata.get("amends_document_id")
            if original_id:
                change_type = metadata.get("change_type", "amends")
                self.pgs.add_change_link(original_id, doc_id, change_type)

        # 6. Загрузка в базу знаний
        self.knowledge.add_document(
            doc_id=doc_id,
            content=content,
            title=metadata.get("title", path.stem) if metadata else path.stem,
            doc_type=doc_type,
            metadata=metadata,
        )

        # 7. Добавление узла в ПГС
        self.pgs.add_document_node(
            doc_id=doc_id,
            title=metadata.get("title", path.stem) if metadata else path.stem,
            doc_type=doc_type,
            metadata=metadata,
        )

        logger.info(f"Документ загружен: {doc_id} [{doc_type}]")
        return {
            "status": "ok",
            "doc_id": doc_id,
            "doc_type": doc_type,
            "questions": None,
        }

    def _parse(self, path: Path) -> str | None:
        """Извлечение текста из файла."""
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                logger.error(f"Ошибка парсинга PDF: {e}")
                return None
        elif suffix == ".docx":
            try:
                from docx import Document
                doc = Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception as e:
                logger.error(f"Ошибка парсинга DOCX: {e}")
                return None
        else:
            logger.warning(f"Неизвестный формат: {suffix}")
            return None

    def _classify(self, content: str, metadata: dict = None) -> str:
        """Классификация типа документа."""
        if metadata and metadata.get("type"):
            return metadata["type"]

        content_lower = content[:2000].lower()

        if any(kw in content_lower for kw in ["федеральный закон", "статья", "кодекс", "постановление"]):
            return DocumentType.LAW
        elif any(kw in content_lower for kw in ["договор", "стороны договорились", "обязуется"]):
            return DocumentType.CONTRACT
        elif any(kw in content_lower for kw in ["правила", "регламент", "порядок"]):
            return DocumentType.REGULATION
        elif any(kw in content_lower for kw in ["инструкция", "руководство"]):
            return DocumentType.INSTRUCTION
        elif any(kw in content_lower for kw in ["изменение", "поправка", "дополнение к"]):
            return DocumentType.AMENDMENT

        return DocumentType.UNKNOWN

    def _generate_id(self, filename: str, doc_type: str) -> str:
        """Генерация уникального ID документа."""
        import hashlib
        import time
        raw = f"{doc_type}:{filename}:{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
