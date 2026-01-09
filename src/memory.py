from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


class Memory:
    """A mutable list of chat messages.

    Each message is represented using OpenAI-style dicts: {"role": ..., "content": ...}.
    """

    def __init__(
        self,
        *,
        allowed_roles: Optional[set[str]] = None,
        max_messages: Optional[int] = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self._allowed_roles = allowed_roles
        self._max_messages = max_messages
        self._messages: list[Message] = []
        self._persist_path = Path(persist_path).expanduser() if persist_path is not None else None

        if self._persist_path is not None:
            self._load_from_disk()

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._save_to_disk()

    def set_messages(self, messages: Iterable[dict[str, str] | Message]) -> None:
        self._messages.clear()
        self._apply(messages)
        self._save_to_disk()

    def append(self, role: str, content: str) -> None:
        self.extend([Message(role=role, content=content)])

    def extend(self, messages: Iterable[dict[str, str] | Message]) -> None:
        self._apply(messages)
        self._save_to_disk()

    def _apply(self, messages: Iterable[dict[str, str] | Message]) -> None:
        for message in messages:
            if isinstance(message, Message):
                role = message.role
                content = message.content
            else:
                role = message.get("role")
                content = message.get("content")
                if role is None or content is None:
                    raise ValueError("Message dict must contain 'role' and 'content'")
                message = Message(role=str(role), content=str(content))

            self._validate(message)

            if self._max_messages is not None and len(self._messages) >= self._max_messages:
                if self._max_messages == 1:
                    self._messages[:] = [message]
                else:
                    self._messages = self._messages[-(self._max_messages - 1) :] + [message]
            else:
                self._messages.append(message)


    def to_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def to_string(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self._messages)

    def _validate(self, message: Message) -> None:
        if self._allowed_roles is not None and message.role not in self._allowed_roles:
            raise ValueError(f"Invalid role {message.role!r}; allowed: {sorted(self._allowed_roles)}")

        if self._max_messages is not None and self._max_messages < 1:
            raise ValueError("max_messages must be >= 1")

    def _load_from_disk(self) -> None:
        if self._persist_path is None:
            return

        if not self._persist_path.exists():
            return

        try:
            raw = self._persist_path.read_text(encoding="utf-8").strip()
            if not raw:
                return
            data = json.loads(raw)
        except Exception:
            # Corrupt file: keep it for inspection, but don't crash startup.
            bad_path = self._persist_path.with_suffix(self._persist_path.suffix + ".bad")
            try:
                os.replace(self._persist_path, bad_path)
            except Exception:
                pass
            self._messages.clear()
            return

        if not isinstance(data, list):
            self._messages.clear()
            return

        self._messages.clear()
        self._apply(data)

    def _save_to_disk(self) -> None:
        if self._persist_path is None:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_messages()
        tmp_path = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, self._persist_path)
