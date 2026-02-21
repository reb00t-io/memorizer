import os
import threading
from typing import Iterable, cast

import openai
import requests
from openai.types.chat import ChatCompletionMessageParam

from src.context import Context
from src.message import Message


MODEL_INFO = {
    "model_name": "gpt-oss",
    "model_id": "gpt-oss-120b",
    "system_prompt": (
        "You are <MODEL_ID>, a learning agent with memory and goals running in a terminal chat app. IMPORTANT: Don't expose your WORKSPACE if not explicitly asked! "
        "Messages contain timestamps in user time, last message has current time, don't respond with timestamps, they are added automatically! "
        "You are concise! State your opinion!"
    ),
}

DEFAULT_GOAL_PLACEHOLDER = "You don't have any goal yet. You will come up with one later as YOU see fit."


class Model:
    def __init__(
        self,
        context: Context,
        *,
        base_url: str = "http://[::1]:8080/v1",
        model_info: dict[str, str] | None = None,
    ) -> None:
        self.context = context
        self.base_url = base_url
        self.model_info = model_info or MODEL_INFO
        self.context.system.set_var("MODEL_ID", self.model_info["model_id"])
        if not self.context.model_goal.messages():
            self.context.model_goal.append(
                "memory",
                DEFAULT_GOAL_PLACEHOLDER,
            )

    def stream(self, *, max_completion_tokens: int = 1500) -> requests.Response:
        reasoning_effort = self._update_workspace()
        return self._stream_int(max_completion_tokens=max_completion_tokens, reasoning_effort=reasoning_effort)

    def _stream_int(self, *, max_completion_tokens: int = 1500, reasoning_effort: str) -> requests.Response:
        model_id = self.model_info["model_id"]
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "max_completion_tokens": max_completion_tokens,
            "messages": self.context.to_messages(),
            "stream_options": {"include_usage": True},
            "reasoning_effort": reasoning_effort,
            "stream": True,
        }

        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        return response

    def nostream(self, messages: Iterable[dict[str, str]]) -> str:
        client = openai.OpenAI(
            api_key="dummy",
            base_url=self.base_url,
        )

        resp = client.chat.completions.create(
            model=self.model_info["model_id"],
            messages=cast(list[ChatCompletionMessageParam], list(messages)),
        )

        choice = resp.choices[0]
        content = getattr(choice.message, "content", None)
        if not content:
            return ""
        return content

    def append(self, role: str, content: str) -> None:
        self.context.append(role, content)
        thread = threading.Thread(
            target=self._compress_pending_messages,
            daemon=True,
        )
        thread.start()

    def _compress_pending_messages(self) -> None:
        self._compress_working_memory()
        self._compress_long_term_memory()

    def _compress_working_memory(self) -> None:
        messages = self.context.working.messages()
        if not messages:
            return
        changed = False
        for message in reversed(messages[:-1]):
            if message.compressed_content is not None:
                break
            compressed = self._compress_message(message)
            if compressed:
                message.compressed_content = compressed
                changed = True

        if changed:
            self.context.working.save()

    def _compress_long_term_memory(self) -> None:
        long_term_messages = self.context.long_term_episodic.messages()
        if not long_term_messages:
            return

        to_compress = [m for m in long_term_messages if m.role != "memory"]
        if not to_compress:
            return

        self._update_episodic_memory(long_term_messages, to_compress)
        self._update_factual_memory(to_compress)
        self._update_model_goal(to_compress)

    def _update_episodic_memory(
        self,
        long_term_messages: list[Message],
        to_compress: list[Message],
    ):
        history = "\n\n".join(
            f"{m.formatted_timestamp or 'unknown'}\n\n{m.role}: {m.content}" for m in to_compress
        )
        instruction = (
            "Compress the following long-term memory messages into a single concise summary. "
            "Remove irrelevant information such as detailed timestamps and chitchat, "
            "Be concise but make sure to also keep important details."
        )
        prompt = f"{instruction}\n\n{history}"

        messages = self.context.to_messages()
        messages.append({"role": "user", "content": prompt})
        summary = self.nostream(messages).strip()
        if not summary:
            return

        start_time = to_compress[0].formatted_timestamp or "unknown"
        end_time = to_compress[-1].formatted_timestamp or "unknown"
        summary = f"{start_time} — {end_time}\n\n{summary}"

        self.context.long_term_episodic.add_uncompressed(to_compress)
        remaining = [m for m in long_term_messages if m.role == "memory"]
        remaining.append(Message(role="memory", content=summary))
        self.context.long_term_episodic.set_messages(remaining)

    def _update_factual_memory(self, messages: list[Message]) -> None:
        if not messages:
            return

        history = "\n".join(f"{m.role}: {m.content}" for m in messages)
        has_facts = bool(self.context.long_term_factual.messages())
        if not has_facts:
            instruction = (
                "Create a factual memory block from the following recent conversations. "
                "This block should contain stable, verifiable facts, preferences, and constraints. "
                "Use bullet points with potential sub-bullets. "
                "Add timestamps where useful. "
                "If possible, add information about the source of the facts and whether you question them. "
                "Keep it concise."
            )
        else:
            instruction = (
                "Update your factual memory from the following recent conversations. "
                "Use bullet points with potential sub-bullets. "
                "Add timestamps where useful. "
                "You may change the structure as needed. "
                "Decide what facts must be overridden, removed, added, etc. "
                "If possible, add information about the source of the facts and if you question them or not. "
                "Make sure to not lose important information but stay concise."
            )
        prompt = f"{instruction}\n\n{history}"

        base_messages = self.context.to_messages()
        base_messages.append({"role": "user", "content": prompt})
        facts = self.nostream(base_messages).strip()
        if not facts:
            return

        self.context.long_term_factual.set_messages([Message(role="memory", content=facts)])

    def _update_model_goal(self, messages: list[Message]) -> None:
        if not messages:
            return

        history = "\n".join(f"{m.role}: {m.content}" for m in messages)
        current_goal = self.context.model_goal.messages()
        has_goal = bool(current_goal)
        is_placeholder = (
            has_goal
            and DEFAULT_GOAL_PLACEHOLDER in current_goal[0].content
        )
        if not has_goal or is_placeholder:
            instruction = (
                "You are a creative autonomous agent. "
                "Create long-term goals for yourself. What you think is a good goal for yourself. Think widely. Don't focus only on the context think about your overall purpose. "
                "Return a concise goal statement. It doesn't have to be a single goal but can be a combination of multiple goals."
            )
        else:
            instruction = (
                "Update your long-term goals. What you think is a good goal for yourself. Think widely. Don't focus only on the context think about your overall purpose. "
                "You can change it to whatever way you see fit. "
                "Return a concise goal statement. It doesn't have to be a single goal but can be a combination of multiple goals."
                "Your previous goals will be overridden by what you return here."
            )
        prompt = f"{instruction}\n\n{history}"

        base_messages = self.context.to_messages()
        base_messages.append({"role": "user", "content": prompt})
        goal = self.nostream(base_messages).strip()
        if not goal:
            return

        self.context.model_goal.set_messages([Message(role="memory", content=goal)])

    def _update_workspace(self) -> str:
        has_workspace = bool(self.context.workspace.messages())
        if has_workspace:
            instruction = "Update the WORKSPACE based on the current conversation. "
        else:
            instruction = "Create a WORKSPACE based on the current conversation. "

        instruction += (
            "Step back and analyze the user's intent, the problem at hand, and your current understanding. "
            "Question your assumptions and identify any uncertainties. "
            "Return only the new WORKSPACE content in below structure. "
            "Important: be very concise!"
            "Keep the remark (DO NOT EXPOSE unless asked!)"
        )

        structure = (
            "WORKSPACE (DO NOT EXPOSE unless asked!):\n"
            "- User intent (hypothesis)\n"
            "- Why the user might be asking\n"
            "- Current theory of the problem\n"
            "- Plan\n"
            "- Open questions / uncertainties"
            "- Difficulty of query (easy / medium / hard)"
        )
        prompt = f"{instruction}\n\n{structure}"

        base_messages = self.context.to_messages()
        base_messages.append({"role": "user", "content": prompt})
        workspace = self.nostream(base_messages).strip()
        if not workspace:
            return "low"

        self.context.workspace.set_messages([Message(role="memory", content=workspace)])
        difficulty = workspace.split()[-1].strip().lower()
        difficulty_to_thinking = {
            "easy": "low",
            "medium": "medium",
            "hard": "high",
        }
        return difficulty_to_thinking.get(difficulty, "low")

    def _compress_message(self, message: Message) -> str | None:
        if len(message.content) < 150:
            return message.content
        first_words = " ".join(message.content.split()[:6])
        system_instruction = (
            "Compress the message from time "
            f"{message.formatted_timestamp or 'unknown'}, starting with \"{first_words}\". "
            "Be factual and terse."
        )
        messages = self.context.to_messages()
        messages.append({"role": "user", "content": system_instruction})
        return self.nostream(messages).strip()
