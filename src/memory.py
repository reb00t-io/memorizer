from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from .message import Message


class Memory:
    """A mutable list of chat messages.

    Each message is represented using OpenAI-style dicts: {"role": ..., "content": ...}.
    """

    def __init__(
        self,
        *,
        allowed_roles: Optional[set[str]] = None,
        max_messages: Optional[int] = None,
        drop_chunk_size: int = 1,
        is_working_memory: bool = False,
        is_short_term_memory: bool = False,
        is_long_term_memory: bool = False,
        render_prefix: str | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self._allowed_roles = allowed_roles
        self._max_messages = max_messages
        self._drop_chunk_size = drop_chunk_size
        self._use_compressed_content = not is_working_memory
        self._include_timestamp = is_working_memory or is_short_term_memory
        self._is_working_memory = is_working_memory
        self._is_long_term_memory = is_long_term_memory
        self._render_prefix = render_prefix
        self._messages: list[Message] = []
        self._uncompressed: list[Message] = []
        self._persist_path = Path(persist_path).expanduser() if persist_path is not None else None

        if self._persist_path is not None:
            self._load_from_disk()

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._save_to_disk()

    def save(self) -> None:
        self._save_to_disk()

    def messages(self) -> list[Message]:
        return list(self._messages)

    def uncompressed(self) -> list[Message]:
        return list(self._uncompressed)

    def add_uncompressed(self, messages: Iterable[Message]) -> None:
        if not self._is_long_term_memory:
            return
        self._uncompressed.extend(messages)
        self._save_to_disk()

    def set_messages(self, messages: Iterable[dict[str, str] | Message]) -> None:
        self._messages.clear()
        self._apply(messages)
        self._save_to_disk()

    def append(self, role: str, content: str) -> list[Message]:
        return self.extend([Message(role=role, content=content)])

    def extend(self, messages: Iterable[dict[str, str] | Message]) -> list[Message]:
        dropped: list[Message] = []
        self._apply(messages, dropped=dropped)
        self._save_to_disk()
        return dropped

    def set_var(self, var: str, value: str) -> None:
        token = f"<{var}>"
        updated: list[Message] = []
        changed = False
        for message in self._messages:
            content = message.content.replace(token, str(value))
            if content != message.content:
                changed = True
                updated.append(
                    Message(
                        role=message.role,
                        content=content,
                        compressed_content=message.compressed_content,
                        timestamp=message.timestamp,
                        formatted_timestamp=message.formatted_timestamp,
                    )
                )
            else:
                updated.append(message)

        if changed:
            self._messages = updated
            self._save_to_disk()

    def _apply(
        self,
        messages: Iterable[dict[str, str] | Message],
        *,
        dropped: Optional[list[Message]] = None,
    ) -> None:
        for message in messages:
            if isinstance(message, Message):
                role = message.role
                content = message.content
            else:
                message = self._parse_message(message)

            if message.timestamp is None:
                message.timestamp = datetime.now(timezone.utc).isoformat()

            if message.formatted_timestamp is None and message.timestamp is not None:
                message.formatted_timestamp = self._format_timestamp(message.timestamp)

            self._validate(message)

            if self._max_messages is not None and len(self._messages) >= self._max_messages:
                drop_size = max(1, self._drop_chunk_size)
                while self._messages and len(self._messages) >= self._max_messages:
                    chunk = self._messages[:drop_size]
                    del self._messages[:drop_size]
                    if dropped is not None:
                        dropped.extend(chunk)

            self._messages.append(message)

    def _parse_message(self, message: dict[str, str]) -> Message:
        role = message.get("role")
        content = message.get("content")
        compressed_content = message.get("compressed_content")
        timestamp = message.get("timestamp")
        if role is None or content is None:
            raise ValueError("Message dict must contain 'role' and 'content'")
        formatted_timestamp = self._format_timestamp(str(timestamp)) if timestamp is not None else None
        return Message(
            role=str(role),
            content=str(content),
            compressed_content=str(compressed_content) if compressed_content is not None else None,
            timestamp=str(timestamp) if timestamp is not None else None,
            formatted_timestamp=formatted_timestamp,
        )

    def _parse_messages(self, messages: Iterable[dict[str, str]]) -> list[Message]:
        return [self._parse_message(message) for message in messages]

    def _render_content(self, m: Message, idx: int) -> str:
        compress = self._use_compressed_content and m.compressed_content is not None
        content = m.compressed_content if compress and m.compressed_content else m.content

        if idx == 0 and self._render_prefix:
            content = f"{self._render_prefix}\n\n{content}"
        if not self._include_timestamp:
            return content

        # no timestamp for the last two messages and current time for the last message
        # (last we don't want the model to start emitting timestamps itself)
        timestamp = None
        if idx < len(self._messages) - 2 and m.formatted_timestamp is not None:
            timestamp = m.formatted_timestamp

        elif idx == len(self._messages) - 1:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        return content if timestamp is None else f"{timestamp}\n\n{content}"

    def _format_timestamp(self, value: str) -> str:
        try:
            parsed = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(parsed)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": self._render_role(message), "content": self._render_content(message, i)}
            for i, message in enumerate(self._messages)
        ]

    def rendered_size_bytes(self) -> int:
        payload = self.to_messages()
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _render_role(self, message: Message) -> str:
        return "system" if message.role == "memory" else message.role

    def _serialize(self) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for message in self._messages:
            item: dict[str, object] = {"role": message.role, "content": message.content}
            if message.compressed_content is not None:
                item["compressed_content"] = message.compressed_content
            if message.timestamp is not None:
                item["timestamp"] = message.timestamp
            payload.append(item)
        if not self._is_long_term_memory:
            return payload

        return [
            {"messages": payload, "uncompressed": self._serialize_uncompressed()},
        ]

    def _serialize_uncompressed(self) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for message in self._uncompressed:
            item: dict[str, object] = {"role": message.role, "content": message.content}
            if message.compressed_content is not None:
                item["compressed_content"] = message.compressed_content
            if message.timestamp is not None:
                item["timestamp"] = message.timestamp
            payload.append(item)
        return payload

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

        if (
            self._is_long_term_memory
            and len(data) == 1
            and isinstance(data[0], dict)
            and "messages" in data[0]
        ):
            payload = data[0]
            raw_messages = payload.get("messages", [])
            raw_uncompressed = payload.get("uncompressed", [])
            self._messages = self._parse_messages(raw_messages)
            self._uncompressed = self._parse_messages(raw_uncompressed)
            return

        self._messages.clear()
        self._apply(data)

    def _save_to_disk(self) -> None:
        if self._persist_path is None:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize()
        tmp_path = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, self._persist_path)
