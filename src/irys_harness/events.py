from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .state import RunState


class EventLogger:
    def __init__(self, state: RunState, verbose: bool = True) -> None:
        self.state = state
        self.verbose = verbose

    def emit(self, label: str, message: str, **fields: Any) -> None:
        event = {
            "ts": datetime.now(UTC).isoformat(),
            "label": label,
            "message": message,
            "fields": fields,
        }
        self.state.events.append(event)
        if self.verbose:
            suffix = " ".join(f"{key}={value}" for key, value in fields.items())
            print(f"[{label}] {message}" + (f" {suffix}" if suffix else ""))

