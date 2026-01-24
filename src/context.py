from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .memory import Memory, Message


@dataclass(slots=True)
class Context:
    """Fixed-layout context composed of separate memory sections.

    Order (fixed):
    1) System message
    2) Long-term memory
    3) Short-term memory
    4) Recall memory (optional; currently empty placeholder)
    5) Working memory
    """

    system: Memory
    long_term: Memory
    short_term: Memory
    recall: Memory
    working: Memory
    layers: list[Memory]

    @classmethod
    def create(
        cls,
        *,
        system_prompt: str = "",
        data_dir: str | Path | None = None,
        persist: bool = True,
        persist_long_term: Optional[bool] = None,
    ) -> "Context":
        base_dir = Path(data_dir).expanduser() if data_dir is not None else (
            Path.home() / ".memorizer")

        system_path = base_dir / "system_memory.json"
        long_term_path = base_dir / "long_term_memory.json"
        short_term_path = base_dir / "short_term_memory.json"
        recall_path = base_dir / "recall_memory.json"
        working_path = base_dir / "working_memory.json"

        persist_system_path: Optional[Path] = system_path if persist else None
        persist_short_term_path: Optional[Path] = short_term_path if persist else None
        persist_recall_path: Optional[Path] = recall_path if persist else None
        persist_working_path: Optional[Path] = working_path if persist else None
        persist_long_term_enabled = persist if persist_long_term is None else bool(
            persist_long_term)
        persist_long_term_path: Optional[Path] = long_term_path if persist_long_term_enabled else None

        system = Memory(allowed_roles={"system"},
                        max_messages=1, persist_path=persist_system_path)
        if system_prompt:
            system.append("system", system_prompt)

        user_assistant = {"user", "assistant"}
        long_term_roles = {"system", "user", "assistant"}
        ctx = cls(
            system=system,
            long_term=Memory(allowed_roles=long_term_roles,
                             persist_path=persist_long_term_path),
            short_term=Memory(allowed_roles=user_assistant,
                              max_messages=30,
                              drop_chunk_size=10,
                              is_short_term_memory=True, persist_path=persist_short_term_path),
            recall=Memory(allowed_roles=user_assistant,
                          persist_path=persist_recall_path),
            working=Memory(
                allowed_roles=user_assistant,
                max_messages=10,
                drop_chunk_size=2,
                is_working_memory=True,
                persist_path=persist_working_path,
            ),
            layers=[],
        )
        ctx.layers = [ctx.working, ctx.short_term, ctx.long_term, ctx.recall]
        return ctx

    def to_messages(self) -> list[dict[str, str]]:
        all_messages = []
        all_messages.extend(self.system.to_messages())
        all_messages.extend(self.long_term.to_messages())
        all_messages.extend(self.short_term.to_messages())
        all_messages.extend(self.working.to_messages())
        return all_messages

    def append(self, role: str, content: str) -> None:
        dropped: list[Message] = self.working.append(role, content)
        for memory in self.layers[1:]:
            if not dropped:
                break
            dropped = memory.extend(dropped)

    def to_string(self) -> str:
        parts: list[str] = []

        parts.append("# System")
        parts.append(self.system.to_string())

        parts.append("\n# Long-term")
        parts.append(self.long_term.to_string())

        parts.append("\n# Short-term")
        parts.append(self.short_term.to_string())

        parts.append("\n# Recall")
        parts.append(self.recall.to_string())

        parts.append("\n# Working")
        parts.append(self.working.to_string())

        return "\n".join(parts).strip() + "\n"
