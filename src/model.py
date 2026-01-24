import os
import threading
from typing import Iterable, cast

import openai
import requests
from openai.types.chat import ChatCompletionMessageParam

from src.context import Context
from src.memory import Message


MODEL_INFO = {
    "model_name": "gpt-oss",
    "model_id": "gpt-oss-120b",
    "system_prompt": "You are <MODEL_ID>, running in a terminal chat app. Messages contain timestamps. The last message has the current time. You are talking very concisely.",
}


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

    def stream(self, *, max_completion_tokens: int = 1500) -> requests.Response:
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
            "reasoning_effort": "low",
            "stream": True,
        }

        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        return response

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
        return

    def _compress_message(self, message: Message) -> str | None:
        if len(message.content) < 150:
            return message.content
        first_words = " ".join(message.content.split()[:6])
        system_instruction = (
            "Compress the messahe from time "
            f"{message.timestamp or 'unknown'}, starting with \"{first_words}\". "
            "Be factual and terse."
        )
        messages = self.context.to_messages()
        messages.append({"role": "user", "content": system_instruction})
        return self.nostream(messages).strip()

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

    def compress(self) -> str:
        short_term_text = self.context.short_term.to_string().strip()
        working_text = self.context.working.to_string().strip()

        combined = "\n".join(
            part for part in [short_term_text, working_text] if part
        ).strip()

        if not combined:
            self.context.short_term.clear()
            self.context.working.clear()
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
        prompt = f"{system_instruction}\n\n{user_content}"

        messages = self.context.to_messages()
        messages.append({"role": "user", "content": prompt})

        summary = self.nostream(messages).strip()

        if summary:
            self.context.long_term.append(
                "system",
                "Long-term memory update (compressed from short-term + working memory):\n"
                + summary,
            )

        self.context.short_term.clear()
        self.context.working.clear()
        return summary
