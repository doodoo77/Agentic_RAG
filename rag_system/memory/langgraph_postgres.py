from __future__ import annotations

from typing import Any, Iterable, Optional

from langgraph.store.postgres import PostgresStore

from rag_system.models.schemas import LongTermMemoryValue


class LangGraphPostgresMemoryClient:
    """Small wrapper around LangGraph's PostgresStore for project-scoped long-term memory."""

    def __init__(self, db_uri: str, *, setup_store: bool = False, scan_limit: int = 1000):
        self.db_uri = db_uri
        self.scan_limit = scan_limit
        self._store_cm = PostgresStore.from_conn_string(db_uri)
        self.store = self._store_cm.__enter__()
        if setup_store:
            self.store.setup()

    def close(self) -> None:
        if getattr(self, '_store_cm', None) is not None:
            self._store_cm.__exit__(None, None, None)
            self._store_cm = None

    @staticmethod
    def namespace(project_id: str) -> tuple[str, str]:
        return ('project_memory', project_id)

    def save_mapping(
        self,
        *,
        project_id: str,
        memory_key: str,
        memory_value: LongTermMemoryValue,
    ) -> None:
        self.store.put(self.namespace(project_id), memory_key, memory_value)

    def list_project_memories(
        self,
        project_id: str,
        *,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        rows = self.store.search(self.namespace(project_id), limit=limit or self.scan_limit)
        return [self._normalize_search_item(row) for row in rows]

    @staticmethod
    def _normalize_search_item(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            key = row.get('key')
            value = row.get('value')
            namespace = row.get('namespace')
        else:
            key = getattr(row, 'key', None)
            value = getattr(row, 'value', None)
            namespace = getattr(row, 'namespace', None)

        if not isinstance(value, dict):
            value = {}

        return {
            'key': str(key or ''),
            'value': value,
            'namespace': tuple(namespace) if isinstance(namespace, Iterable) and not isinstance(namespace, str) else namespace,
        }
