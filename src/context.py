from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .memory import Memory
from .message import Message


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
    model_goal: Memory
    workspace: Memory
    long_term_episodic: Memory
    long_term_factual: Memory
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
        model_goal_path = base_dir / "model_goal.json"
        workspace_path = base_dir / "workspace.json"
        long_term_path = base_dir / "ltm_episodic.json"
        long_term_factual_path = base_dir / "ltm_factual.json"
        short_term_path = base_dir / "stm.json"
        recall_path = base_dir / "recall_memory.json"
        working_path = base_dir / "wm.json"

        persist_system_path: Optional[Path] = system_path if persist else None
        persist_model_goal_path: Optional[Path] = model_goal_path if persist else None
        persist_workspace_path: Optional[Path] = workspace_path if persist else None
        persist_short_term_path: Optional[Path] = short_term_path if persist else None
        persist_recall_path: Optional[Path] = recall_path if persist else None
        persist_working_path: Optional[Path] = working_path if persist else None
        persist_long_term_enabled = persist if persist_long_term is None else bool(
            persist_long_term)
        persist_long_term_path: Optional[Path] = long_term_path if persist_long_term_enabled else None
        persist_long_term_factual_path: Optional[Path] = (
            long_term_factual_path if persist_long_term_enabled else None
        )

        system = Memory(allowed_roles={"system"},
                        max_messages=1, persist_path=persist_system_path)
        if system_prompt:
            system.append("system", system_prompt)

        user_assistant = {"user", "assistant"}
        long_term_roles = {"system", "user", "assistant", "memory"}
        goal_roles = {"memory"}
        ctx = cls(
            system=system,
            model_goal=Memory(allowed_roles=goal_roles,
                              max_messages=1,
                              render_prefix="#Assistant goals",
                              persist_path=persist_model_goal_path),
            workspace=Memory(allowed_roles=goal_roles,
                             max_messages=1,
                             render_prefix="#Workspace (DO NOT expose unless asked!)",
                             persist_path=persist_workspace_path),
            long_term_episodic=Memory(allowed_roles=long_term_roles,
                             is_long_term_memory=True,
                             render_prefix="#Long-term episodic memory",
                             persist_path=persist_long_term_path),
            long_term_factual=Memory(allowed_roles=long_term_roles,
                             max_messages=1,
                             is_long_term_memory=True,
                             render_prefix="#Long-term factual memory",
                             persist_path=persist_long_term_factual_path),
            short_term=Memory(allowed_roles=user_assistant,
                              max_messages=30,
                              drop_chunk_size=10,
                              render_prefix="#short-term memory",
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
        ctx.layers = [ctx.working, ctx.short_term, ctx.long_term_episodic]
        return ctx

    def to_messages(self) -> list[dict[str, str]]:
        all_messages = []
        # system message is usually stable so put it first
        # long_term_episodic is also stable at the start
        # end of long_term_episodic changes at the same time as model_goal and long_term_factual
        all_messages.extend(self.system.to_messages())
        all_messages.extend(self.long_term_episodic.to_messages())
        all_messages.extend(self.model_goal.to_messages())
        all_messages.extend(self.long_term_factual.to_messages())

        # frequently changing short term goals, but still a bit stable
        all_messages.extend(self.short_term.to_messages())
        # recall changes most frequently but we have to put it before working memory
        # as the model must see the recent conversation in the end
        all_messages.extend(self.workspace.to_messages())
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

        parts.append("\n# Model goal")
        parts.append(self.model_goal.to_string())

        parts.append("\n# Workspace")
        parts.append(self.workspace.to_string())

        parts.append("\n# Long-term episodic")
        parts.append(self.long_term_episodic.to_string())

        parts.append("\n# Long-term factual")
        parts.append(self.long_term_factual.to_string())

        parts.append("\n# Short-term")
        parts.append(self.short_term.to_string())

        parts.append("\n# Recall")
        parts.append(self.recall.to_string())

        parts.append("\n# Working")
        parts.append(self.working.to_string())

        return "\n".join(parts).strip() + "\n"
