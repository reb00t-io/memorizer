
import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from . import config
from .completion import stream_completion
from ..model import Model


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
        f"{model.model_name}. "
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
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="memorizer-chat",
        description="Interactive Memorizer chat REPL against an OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MEMORIZER_BASE_URL", config.BASE_URL),
        help="Endpoint base URL (env: MEMORIZER_BASE_URL).",
    )
    parser.add_argument(
        "--model",
        dest="model_id",
        default=os.environ.get("MEMORIZER_MODEL_ID", config.MODEL_ID),
        help="Model id sent to the endpoint (env: MEMORIZER_MODEL_ID).",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MEMORIZER_MODEL_NAME", config.MODEL_NAME),
        help="Display name shown in the banner (env: MEMORIZER_MODEL_NAME).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MEMORIZER_API_KEY", "dummy"),
        help="API key; defaults to 'dummy' for local servers (env: MEMORIZER_API_KEY).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=int(os.environ.get("MEMORIZER_MAX_COMPLETION_TOKENS", config.MAX_COMPLETION_TOKENS)),
        help="Max completion tokens per response (env: MEMORIZER_MAX_COMPLETION_TOKENS).",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=os.environ.get("MEMORIZER_THINKING", "").lower() in ("1", "true", "yes"),
        help="Enable model reasoning/thinking; off by default for lower latency "
        "(env: MEMORIZER_THINKING).",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        default=os.environ.get("MEMORIZER_MEMORY", "").lower() in ("1", "true", "yes"),
        help="Enable Qdrant-backed recall memory + the `recall` tool "
        "(needs the 'store' extra; env: MEMORIZER_MEMORY).",
    )
    parser.add_argument(
        "--org",
        action="store_true",
        default=os.environ.get("MEMORIZER_ORG", "").lower() in ("1", "true", "yes"),
        help="Also maintain shared organization memory (implies --memory).",
    )
    parser.add_argument(
        "--org-profile",
        default=os.environ.get("MEMORIZER_ORG_PROFILE"),
        help="Path to the org profile / extraction-rules doc (env: MEMORIZER_ORG_PROFILE).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("MEMORIZER_QDRANT_URL"),
        help="Qdrant server URL; omit for a local on-disk store (env: MEMORIZER_QDRANT_URL).",
    )
    parser.add_argument(
        "--member-id",
        default=os.environ.get("MEMORIZER_MEMBER_ID"),
        help="Member id; scopes personal memory to this member (env: MEMORIZER_MEMBER_ID).",
    )
    parser.add_argument(
        "--role",
        default=os.environ.get("MEMORIZER_ROLE"),
        help="Member role; gates org memory read visibility / write access (env: MEMORIZER_ROLE).",
    )
    parser.add_argument(
        "--org-roles",
        default=os.environ.get("MEMORIZER_ORG_ROLES"),
        help="Comma-separated list of all known org roles (env: MEMORIZER_ORG_ROLES).",
    )
    parser.add_argument(
        "--org-writer-roles",
        default=os.environ.get("MEMORIZER_ORG_WRITER_ROLES"),
        help="Comma-separated roles allowed to write org memory (env: MEMORIZER_ORG_WRITER_ROLES).",
    )
    args = parser.parse_args()

    def _csv(value: str | None) -> list[str] | None:
        if not value:
            return None
        return [v.strip() for v in value.split(",") if v.strip()]

    model = Model.create(
        model_id=args.model_id,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=config.SYSTEM_PROMPT,
        max_completion_tokens=args.max_completion_tokens,
        thinking=args.thinking,
        enable_memory=args.memory or args.org,
        enable_org=args.org,
        org_profile=args.org_profile,
        org_roles=_csv(args.org_roles),
        org_writer_roles=_csv(args.org_writer_roles),
        member_id=args.member_id,
        role=args.role,
        qdrant_location=args.qdrant_url,
    )
    asyncio.run(_chat_loop(
        model, max_completion_tokens=args.max_completion_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
