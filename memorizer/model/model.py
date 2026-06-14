import json
import threading
from pathlib import Path
from typing import Iterable, Optional

import requests

from .context import Context
from .message import Message


def process_streaming_response(response: requests.Response) -> tuple[str, list[dict]]:
    """Process a streaming chat completions response (SSE format).

    Returns (assistant_text, tool_calls) where tool_calls are accumulated from
    incremental delta chunks and returned in the OpenAI tool call format.

    Always closes the response when done (or on error) to release the
    underlying TCP connection back to the pool.
    """
    text_parts: list[str] = []
    tool_calls_map: dict[int, dict] = {}

    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                text_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    tool_calls_map[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_calls_map[idx]["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]
    finally:
        response.close()

    text = "".join(text_parts).strip()
    tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())]
    return text, tool_calls


DEFAULT_BASE_URL = "http://[::1]:8080/v1"

DEFAULT_GOAL_PLACEHOLDER = "You don't have any goal yet. You will come up with one later as YOU see fit."


class Model:
    """Model-managed context client for an OpenAI-compatible chat endpoint.

    The endpoint, model id and completion-token budget are supplied at startup
    so the class can be driven directly from another project without the bundled
    chat CLI. Use :meth:`create` to build the backing :class:`Context` in one call,
    or pass an existing context to ``__init__``.
    """

    def __init__(
        self,
        context: Context,
        *,
        model_id: str,
        max_completion_tokens: int,
        model_name: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "dummy",
    ) -> None:
        self.context = context
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.base_url = base_url
        self.api_key = api_key
        self.max_completion_tokens = max_completion_tokens
        self._compression_lock = threading.Lock()
        self.context.system.set_var("MODEL_ID", model_id)
        if not self.context.model_goal.messages():
            self.context.model_goal.append(
                "memory",
                DEFAULT_GOAL_PLACEHOLDER,
            )

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        max_completion_tokens: int,
        system_prompt: str = "",
        model_name: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "dummy",
        data_dir: str | Path | None = None,
        persist: bool = True,
        persist_long_term: Optional[bool] = None,
    ) -> "Model":
        """Build a :class:`Context` and wrap it in a ready-to-use :class:`Model`."""
        context = Context.create(
            system_prompt=system_prompt,
            data_dir=data_dir,
            persist=persist,
            persist_long_term=persist_long_term,
        )
        return cls(
            context,
            model_id=model_id,
            max_completion_tokens=max_completion_tokens,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )

    def stream(
        self,
        messages: list[dict] | None = None,
        *,
        tools: list[dict] | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> requests.Response:
        """Streaming LLM call. Returns a raw requests.Response for SSE processing.

        If messages is None, uses self.context.to_messages().
        If max_completion_tokens is None, uses self.max_completion_tokens.
        If reasoning_effort is None, computes it via _update_workspace().
        """
        if max_completion_tokens is None:
            max_completion_tokens = self.max_completion_tokens
        if reasoning_effort is None:
            reasoning_effort = self._update_workspace()
        if messages is None:
            messages = self.context.to_messages()
        return self._stream_int(
            messages,
            tools=tools,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )

    def stream_and_process(
        self,
        messages: list[dict] | None = None,
        *,
        tools: list[dict] | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Stream a completion and process the SSE response in one step."""
        response = self.stream(
            messages,
            tools=tools,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        return process_streaming_response(response)

    def _stream_int(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_completion_tokens: int,
        reasoning_effort: str,
    ) -> requests.Response:
        model_id = self.model_id
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "max_completion_tokens": max_completion_tokens,
            "messages": messages,
            "stream_options": {"include_usage": True},
            "reasoning_effort": reasoning_effort,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        return response

    def _nostream_int(
        self,
        messages: Iterable[dict],
        *,
        tools: list[dict] | None = None,
        max_completion_tokens: int,
        reasoning_effort: str = "low",
    ) -> dict:
        """Non-streaming LLM call. Returns the raw chat completions response dict."""
        model_id = self.model_id
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": model_id,
            "max_completion_tokens": max_completion_tokens,
            "messages": list(messages),
            "reasoning_effort": reasoning_effort,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def nostream(self, messages: Iterable[dict]) -> str:
        """Convenience wrapper: call nostream() and return just the text content."""
        data = self._nostream_int(messages, max_completion_tokens=self.max_completion_tokens)
        return (data["choices"][0]["message"].get("content") or "").strip()

    def append(self, role: str, content: str) -> None:
        self.context.append(role, content)
        if self._compression_lock.acquire(blocking=False):
            thread = threading.Thread(
                target=self._compress_pending_messages,
                daemon=True,
            )
            thread.start()

    def _compress_pending_messages(self) -> None:
        try:
            self._compress_working_memory()
            self._compress_long_term_memory()
        finally:
            self._compression_lock.release()

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
        summary = self.nostream(messages)
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
        facts = self.nostream(base_messages)
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
        goal = self.nostream(base_messages)
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
        workspace = self.nostream(base_messages)
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
        return self.nostream(messages)
