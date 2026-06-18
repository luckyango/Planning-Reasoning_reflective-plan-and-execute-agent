from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def normalize_string_list(value: Any) -> list[str]:
    """Convert model-provided list-like values into clean strings."""

    if not isinstance(value, list):
        return []

    return [str(item).strip() for item in value if str(item).strip()]


def clamp_score(value: Any) -> float:
    """Normalize a model-provided score into the 0.0 to 1.0 range."""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, score))


def normalize_bool(value: Any) -> bool:
    """Convert model-provided boolean-like values into a real boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)
