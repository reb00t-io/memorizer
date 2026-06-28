import json
import time
import asyncio

from . import config
from ..model import Model


def _consume_stream(response) -> tuple[str, list[dict], dict | None]:
    """Print a streamed SSE response and return (text, tool_calls, usage)."""
    text_parts: list[str] = []
    tool_calls_map: dict[int, dict] = {}
    usage = None
    reasoning = False

    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)

        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        if delta.get("content"):
            if reasoning:
                reasoning = False
                print("\033[0m\n")
            text_parts.append(delta["content"])
            print(delta["content"], end="", flush=True)

        if delta.get("reasoning_content"):
            if not reasoning:
                print("\033[34m")
                reasoning = True
            print(delta["reasoning_content"], end="", flush=True)

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls_map.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function", {})
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]

    print("\033[0m", end="", flush=True)
    text = "".join(text_parts).strip()
    tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]
    return text, tool_calls, usage


async def stream_completion(model: Model, max_completion_tokens: int | None = None):
    """
    Stream an LLM completion, resolving ``recall`` tool calls in a loop.

    Args:
        model: Model instance containing request config and context
        max_completion_tokens: Maximum tokens to generate; falls back to the
            model's configured ``max_completion_tokens`` when None.

    Returns:
        dict: Usage statistics if available, None otherwise
    """
    t_start = time.time()
    usage = None
    tools = model.recall_tools()
    messages = model.context.to_messages()

    try:
        assistant_text = ""
        for _ in range(max(1, model.recall_max_rounds)):
            response = model.stream(
                messages, tools=tools, max_completion_tokens=max_completion_tokens
            )
            assistant_text, tool_calls, turn_usage = _consume_stream(response)
            usage = turn_usage or usage
            if not tool_calls:
                break
            # Tool round: record the call + results, then let the model continue.
            print(f"\n\033[90m[recall: {len(tool_calls)} call(s)]\033[0m", flush=True)
            messages.append(
                {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls}
            )
            messages.extend(model.execute_tool_calls(tool_calls))

        t = time.time() - t_start
        print("\n\n")
        if usage:
            token_count = usage.get("completion_tokens", 0)
            _tpot = t / token_count if token_count > 0 else 0

        if assistant_text:
            model.append("assistant", assistant_text)
            # Refresh the WORKSPACE in the background now that the turn (incl. the
            # model response) is complete — keeps it off the request path.
            model.update_workspace_async()

        return usage

    except Exception as e:
        print(f"Error in streaming: {e}")
        return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stream a test chat completion")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Tell me a joke.",
        help="User prompt to send",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=config.MAX_COMPLETION_TOKENS,
        help="Maximum completion tokens",
    )
    args = parser.parse_args()

    model = Model.create(
        model_id=config.MODEL_ID,
        model_name=config.MODEL_NAME,
        base_url=config.BASE_URL,
        system_prompt=config.SYSTEM_PROMPT,
        max_completion_tokens=args.max_completion_tokens,
    )
    model.context.append("user", args.prompt)
    asyncio.run(stream_completion(model, max_completion_tokens=args.max_completion_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
