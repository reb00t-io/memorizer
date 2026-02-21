
import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from src.chat.completion import stream_completion
from src.model.model import MODEL_INFO, Model
from src.model.context import Context


DEFAULT_MAX_COMPLETION_TOKENS = 8000


def _nice_prompt() -> FormattedText:
    return FormattedText([
        ("class:prompt", "You"),
        ("", "> "),
    ])


def _history_file() -> Path:
    path = Path.home() / ".memorizer" / "chat_history.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def _chat_loop(model: Model, *, max_completion_tokens: int) -> None:
    print(
        f"{MODEL_INFO['model_name']}. "
        "Ctrl-D/Ctrl-C to exit.\n"
    )

    style = Style.from_dict({"prompt": "ansicyan bold"})
    session = PromptSession(history=FileHistory(
        str(_history_file())), style=style)

    while True:
        try:
            user_text = await session.prompt_async(_nice_prompt(), multiline=False)
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        user_text = (user_text or "").strip()
        if not user_text:
            continue

        model.append("user", user_text)
        usage = await stream_completion(model, max_completion_tokens=max_completion_tokens)
        details = usage.get("prompt_tokens_details", None) if usage else None
        prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
        cached_tokens = details.get("cached_tokens", 0) if details else 0
        cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
        sizes = model.context.memory_sizes_bytes()
        total_size = sum(sizes.values())
        sizes_pct = " ".join(
            f"{name[0]}:{(size / total_size * 100):.0f}%"
            for name, size in sizes.items()
            if total_size > 0
        )
        print(f"\n[{prompt_tokens} tokens, {cache_pct:.0f}% cached, {sizes_pct}]\n")


def main() -> int:
    context = Context.create(system_prompt=MODEL_INFO["system_prompt"])
    model = Model(context)
    asyncio.run(_chat_loop(
        model, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
