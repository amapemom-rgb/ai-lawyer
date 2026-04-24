"""
Понятийно-Графовая Система (ПГС).

Сердце архитектуры ИИ-юриста.
Хранит связи между понятиями, документами, прецедентами.
Поддерживает "пузыри" — эмерджентные экосистемы вокруг популярных тем.
"""

from neo4j import GraphDatabase
from loguru import logger


class PGSGraph:
    """Понятийно-графовая система на базе Neo4j."""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self._driver = None
        if uri:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"ПГС подключена к Neo4j: {uri}")

    def close(self):
        if self._driver:
            self._driver.close()

    # === ПОИСК ===

    def search_relevant(self, query: str, limit: int = 20) -> list[dict]:
        """
        Поиск релевантных узлов в графе по запросу.
        Использует полнотекстовый поиск + обход связей.
        """
        if not self._driver:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('concept_index', $query)
                YIELD node, score
                WHERE score > 0.5
                WITH node, score
                ORDER BY score DESC
                LIMIT $limit
                OPTIONAL MATCH (node)-[r]->(related)
                RETURN node, collect({
                    type: type(r),
                    target: related.title,
                    target_type: labels(related)[0]
                }) as connections, score
                """,
                query=query, limit=limit,
            )
            return [
                {
                    "title": record["node"]["title"],
                    "type": record["node"].labels.pop() if record["node"].labels else "Unknown",
                    "properties": dict(record["node"]),
                    "connections": record["connections"],
                    "score": record["score"],
                }
                for record in result
            ]

    # === ОБНОВЛЕНИЕ ===

    def update_from_interaction(
        self, query: str, response: str, documents_used: list
    ):
        """
        Обновление графа после взаимодействия с пользователем.
        Создаёт узлы прецедентов и связи с использованными документами.
        """
        if not self._driver:
            logger.warning("ПГС не подключена — пропускаю обновление")
            return

        with self._driver.session() as session:
            session.run(
                """
                CREATE (i:Interaction {
                    query: $query,
                    response_summary: $response_summary,
                    created_at: datetime()
                })
                WITH i
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})
                CREATE (i)-[:USED_DOCUMENT]->(d)
                """,
                query=query,
                response_summary=response[:500],
                doc_ids=[d.get("id") for d in documents_used if d.get("id")],
            )
            logger.info("ПГС обновлена: добавлен прецедент взаимодействия")

    # === ПУЗЫРИ ===

    def detect_bubbles(self, threshold: int = 5) -> list[dict]:
        """
        Обнаружение "пузырей" — тем, которые набирают популярность
        и заслуживают выделения в отдельную экосистему.
        """
        if not self._driver:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Concept)<-[:RELATES_TO]-(i:Interaction)
                WITH c, count(i) as interaction_count
                WHERE interaction_count >= $threshold
                OPTIONAL MATCH (c)-[:PART_OF]->(b:Bubble)
                WHERE b IS NULL
                RETURN c.title as topic,
                       interaction_count,
                       c.id as concept_id
                ORDER BY interaction_count DESC
                """,
                threshold=threshold,
            )
            bubbles = [
                {
                    "topic": record["topic"],
                    "interaction_count": record["interaction_count"],
                    "concept_id": record["concept_id"],
                }
                for record in result
            ]

            if bubbles:
                logger.info(
                    f"Обнаружено {len(bubbles)} потенциальных пузырей: "
                    f"{', '.join(b['topic'] for b in bubbles)}"
                )
            return bubbles

    def create_bubble(self, topic: str, concept_ids: list[str]) -> str:
        """
        Создание "пузыря" — выделенной экосистемы вокруг темы.
        Возвращает ID созданного пузыря.
        """
        if not self._driver:
            return ""

        with self._driver.session() as session:
            result = session.run(
                """
                CREATE (b:Bubble {
                    id: randomUUID(),
                    title: $topic,
                    created_at: datetime(),
                    status: 'active'
                })
                WITH b
                UNWIND $concept_ids AS cid
                MATCH (c:Concept {id: cid})
                CREATE (c)-[:PART_OF]->(b)
                RETURN b.id as bubble_id
                """,
                topic=topic, concept_ids=concept_ids,
            )
            record = result.single()
            bubble_id = record["bubble_id"] if record else ""
            logger.info(f"Создан пузырь '{topic}' | id={bubble_id}")
            return bubble_id

    # === ЗАГРУЗКА ДОКУМЕНТОВ ===

    def add_document_node(self, doc_id: str, title: str, doc_type: str, metadata: dict = None):
        """Добавление узла документа в граф."""
        if not self._driver:
            return

        with self._driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title = $title,
                    d.type = $doc_type,
                    d.updated_at = datetime()
                """,
                doc_id=doc_id, title=title, doc_type=doc_type,
            )

    def add_change_link(self, original_doc_id: str, new_doc_id: str, change_type: str):
        """
        Добавление связи изменения между документами.
        change_type: 'amends' | 'replaces' | 'supplements' | 'revokes'
        """
        if not self._driver:
            return

        with self._driver.session() as session:
            session.run(
                """
                MATCH (orig:Document {id: $original_id})
                MATCH (new:Document {id: $new_id})
                CREATE (new)-[:CHANGES {
                    type: $change_type,
                    created_at: datetime()
                }]->(orig)
                """,
                original_id=original_doc_id,
                new_id=new_doc_id,
                change_type=change_type,
            )
            logger.info(
                f"ПГС: {new_doc_id} --[{change_type}]--> {original_doc_id}"
            )
