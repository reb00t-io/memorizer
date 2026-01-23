
import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from src.completion import MODEL_INFO, stream_completion
from src.context import Context


DEFAULT_SYSTEM_PROMPT = "You are <MODEL_NAME>, running in a terminal chat app. You are talking very concisely."
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


async def _chat_loop(context: Context, *, max_completion_tokens: int) -> None:
    print(
        f"{MODEL_INFO['model_name']}. "
        "Ctrl-D/Ctrl-C to exit.\n"
    )

    context.system.set_var("MODEL_NAME", MODEL_INFO["model_name"])
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

        context.append("user", user_text)
        await stream_completion(context, max_completion_tokens=max_completion_tokens)
        print("\n")


def main() -> int:
    context = Context.create(system_prompt=DEFAULT_SYSTEM_PROMPT)
    asyncio.run(_chat_loop(
        context, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
