from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from .state import RunState


class EventLogger:
    def __init__(
        self,
        state: RunState,
        verbose: bool = True,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.state = state
        self.verbose = verbose
        self.event_callback = event_callback

    def emit(self, label: str, message: str, **fields: Any) -> None:
        event = {
            "ts": datetime.now(UTC).isoformat(),
            "label": label,
            "message": message,
            "fields": fields,
        }
        self.state.events.append(event)
        if self.event_callback:
            self.event_callback(event)
        if self.verbose:
            suffix = " ".join(f"{key}={value}" for key, value in fields.items())
            print(f"[{label}] {message}" + (f" {suffix}" if suffix else ""))
