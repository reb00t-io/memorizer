from __future__ import annotations

import os
from typing import Iterable

import openai


def llm_nostream(messages: Iterable[dict[str, str]]) -> str:
	"""Call the local OpenAI-compatible endpoint and return assistant text."""

	client = openai.OpenAI(
		api_key=os.environ.get("PRIVATE_MODE_API_KEY"),
		base_url="http://[::1]:8080/v1",
	)

	resp = client.chat.completions.create(
		model="gpt-oss-120b",
		messages=list(messages),
	)

	choice = resp.choices[0]
	content = getattr(choice.message, "content", None)
	if not content:
		return ""
	return content
