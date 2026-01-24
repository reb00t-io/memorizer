import json
import time
import asyncio

from src.context import Context
from src.model import MODEL_INFO, Model


async def stream_completion(model: Model, max_completion_tokens: int = 1500):
    """
    Stream LLM completion using a local OpenAI-compatible endpoint.

    Args:
        model: Model instance containing request config and context
        max_completion_tokens: Maximum tokens to generate (default: 1500)

    Returns:
        dict: Usage statistics if available, None otherwise
    """
    t_start = time.time()
    usage = None
    assistant_text_parts: list[str] = []

    try:
        response = model.stream(max_completion_tokens=max_completion_tokens)

        # Stream and process response
        reasoning = False

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    chunk = json.loads(data)

                    if chunk.get('usage'):
                        usage = chunk['usage']
                    if chunk.get('choices') and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0]['delta']
                        content = None

                        # Assistant visible text
                        if 'content' in delta and delta['content']:
                            if reasoning:
                                reasoning = False
                                print("\033[0m\n")
                            content = delta['content']
                            assistant_text_parts.append(content)

                        # Reasoning tokens (don't rely on model_extra)
                        if 'reasoning_content' in delta and delta['reasoning_content']:
                            if not reasoning:
                                print("\033[34m")
                                reasoning = True
                            content = delta['reasoning_content']

                        if content:
                            print(content, end="", flush=True)

        print("\033[0m", end="", flush=True)

        # Calculate and display statistics
        t = time.time() - t_start
        print("\n\n")
        if usage:
            token_count = usage.get('completion_tokens', 0)
            tpot = t / token_count if token_count > 0 else 0
            #print(f"n={token_count}, t={t:.2f}s, tpot={tpot * 1000:.2f}ms")

        assistant_text = "".join(assistant_text_parts).strip()
        if assistant_text:
            model.append("assistant", assistant_text)

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
        default=1500,
        help="Maximum completion tokens",
    )
    args = parser.parse_args()

    ctx = Context.create(system_prompt=MODEL_INFO["system_prompt"])
    ctx.system.set_var("MODEL_ID", MODEL_INFO["model_id"])
    ctx.append("user", args.prompt)
    model = Model(ctx)
    asyncio.run(stream_completion(model, max_completion_tokens=args.max_completion_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
