from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.utils import utc_now


@dataclass
class MemoryItem:
    """A single working-memory entry created during a run."""

    category: str
    content: str
    source_step_id: int | None = None
    importance: float = 0.5
    created_at: str = field(default_factory=utc_now)


@dataclass
class WorkingMemory:
    """Explicit working memory for multi-step reasoning."""

    observations: list[MemoryItem] = field(default_factory=list)
    decisions: list[MemoryItem] = field(default_factory=list)
    failed_attempts: list[MemoryItem] = field(default_factory=list)
    lessons: list[MemoryItem] = field(default_factory=list)

    def add_observation(
        self,
        content: str,
        source_step_id: int | None = None,
        importance: float = 0.5,
    ) -> None:
        self.observations.append(
            MemoryItem(
                category="observation",
                content=content,
                source_step_id=source_step_id,
                importance=importance,
            )
        )

    def add_decision(
        self,
        content: str,
        source_step_id: int | None = None,
        importance: float = 0.5,
    ) -> None:
        self.decisions.append(
            MemoryItem(
                category="decision",
                content=content,
                source_step_id=source_step_id,
                importance=importance,
            )
        )

    def add_failed_attempt(
        self,
        content: str,
        source_step_id: int | None = None,
        importance: float = 0.8,
    ) -> None:
        self.failed_attempts.append(
            MemoryItem(
                category="failed_attempt",
                content=content,
                source_step_id=source_step_id,
                importance=importance,
            )
        )

    def add_lesson(
        self,
        content: str,
        source_step_id: int | None = None,
        importance: float = 0.8,
    ) -> None:
        self.lessons.append(
            MemoryItem(
                category="lesson",
                content=content,
                source_step_id=source_step_id,
                importance=importance,
            )
        )

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "observations": self._items_to_dict(self.observations),
            "decisions": self._items_to_dict(self.decisions),
            "failed_attempts": self._items_to_dict(self.failed_attempts),
            "lessons": self._items_to_dict(self.lessons),
        }

    def _items_to_dict(self, items: list[MemoryItem]) -> list[dict[str, Any]]:
        return [
            {
                "category": item.category,
                "content": item.content,
                "source_step_id": item.source_step_id,
                "importance": item.importance,
                "created_at": item.created_at,
            }
            for item in items
        ]
