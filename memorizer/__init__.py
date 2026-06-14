"""Memorizer: a structured, long-term memory architecture for LLMs.

Library entry point. The core classes are re-exported here so consumers can:

    from memorizer import Model, Context

    model = Model.create(
        model_id="gpt-oss-120b",
        base_url="http://host:8080/v1",
        system_prompt="You are <MODEL_ID>.",
        max_completion_tokens=1500,
    )

The interactive chat CLI lives in ``memorizer.chat`` and requires the optional
``chat`` extra (``pip install memorizer[chat]``).
"""

from .model import (
    Context,
    DEFAULT_BASE_URL,
    DEFAULT_GOAL_PLACEHOLDER,
    Memory,
    Message,
    Model,
    process_streaming_response,
)

__all__ = [
    "Context",
    "Memory",
    "Message",
    "Model",
    "DEFAULT_BASE_URL",
    "DEFAULT_GOAL_PLACEHOLDER",
    "process_streaming_response",
]
