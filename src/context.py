from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .memory import Memory


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

	@classmethod
	def create(
		cls,
		*,
		system_prompt: str = "",
		data_dir: str | Path | None = None,
		persist: bool = True,
		persist_long_term: Optional[bool] = None,
	) -> "Context":
		base_dir = Path(data_dir).expanduser() if data_dir is not None else (Path.home() / ".memorizer")

		system_path = base_dir / "system_memory.json"
		long_term_path = base_dir / "long_term_memory.json"
		short_term_path = base_dir / "short_term_memory.json"
		recall_path = base_dir / "recall_memory.json"
		working_path = base_dir / "working_memory.json"

		persist_system_path: Optional[Path] = system_path if persist else None
		persist_short_term_path: Optional[Path] = short_term_path if persist else None
		persist_recall_path: Optional[Path] = recall_path if persist else None
		persist_working_path: Optional[Path] = working_path if persist else None
		persist_long_term_enabled = persist if persist_long_term is None else bool(persist_long_term)
		persist_long_term_path: Optional[Path] = long_term_path if persist_long_term_enabled else None

		system = Memory(allowed_roles={"system"}, max_messages=1, persist_path=persist_system_path)
		if system_prompt:
			system.append("system", system_prompt)

		user_assistant = {"user", "assistant"}
		long_term_roles = {"system", "user", "assistant"}
		return cls(
			system=system,
			long_term=Memory(allowed_roles=long_term_roles, persist_path=persist_long_term_path),
			short_term=Memory(allowed_roles=user_assistant, persist_path=persist_short_term_path),
			recall=Memory(allowed_roles=user_assistant, persist_path=persist_recall_path),
			working=Memory(allowed_roles=user_assistant, persist_path=persist_working_path),
		)

	def to_messages(self) -> list[dict[str, str]]:
		messages: list[dict[str, str]] = []
		messages.extend(self.system.to_messages())
		messages.extend(self.long_term.to_messages())
		messages.extend(self.short_term.to_messages())
		messages.extend(self.recall.to_messages())
		messages.extend(self.working.to_messages())
		return messages

	def append(self, role: str, content: str) -> None:
		self.working.append(role, content)

	def compress(self) -> str:
		"""Compress short-term + working memory into long-term memory.

		- Concats all messages from short-term and working into a single string.
		- Calls llm_nostream with a system instruction to summarize concisely.
		- Appends a single system message to long-term explaining what it is.
		- Clears short-term and working memory.
		"""
		from src.llm_nostream import llm_nostream

		short_term_text = self.short_term.to_string().strip()
		working_text = self.working.to_string().strip()

		combined = "\n".join(
			part for part in [short_term_text, working_text] if part
		).strip()

		if not combined:
			self.short_term.clear()
			self.working.clear()
			return ""

		system_instruction = (
			"You are a memory compressor. Extract the most important information "
			"from the conversation for long-term memory. Be concise, stable, and factual. "
			"Prefer bullets. Include only enduring facts, preferences, decisions, and goals."
		)
		user_content = (
			"Compress the following messages into a concise long-term memory summary:\n\n"
			f"{combined}"
		)

		summary = llm_nostream(
			[
				{"role": "system", "content": system_instruction},
				{"role": "user", "content": user_content},
			]
		).strip()

		if summary:
			self.long_term.append(
				"system",
				"Long-term memory update (compressed from short-term + working memory):\n"
				+ summary,
			)

		self.short_term.clear()
		self.working.clear()
		return summary

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
