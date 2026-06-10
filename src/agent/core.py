"""
ИИ-Агент: ядро системы на LangGraph.

Граф состояний (реально исполняется через compiled.invoke()):

    retrieve (поиск в ПГС + базе знаний)
        → pochemuchka (нужны ли уточнения?)
            → [да] → clarify → END
            → [нет] → generate (LLM)
                → validate
                    → [не прошёл, retry < 2] → generate
                    → [прошёл / лимит] → finalize → END
"""
