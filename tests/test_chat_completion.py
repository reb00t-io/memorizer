import asyncio
import json

from memorizer.chat.completion import stream_completion


class _FakeResponse:
    def __init__(self, payload_lines: list[str]) -> None:
        self._payload_lines = payload_lines

    def iter_lines(self):
        for line in self._payload_lines:
            yield line.encode("utf-8")


class _FakeModel:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.append_calls: list[tuple[str, str]] = []
        self.workspace_updates = 0

    def stream(self, *, max_completion_tokens: int = 1500):
        return self._response

    def append(self, role: str, content: str) -> None:
        self.append_calls.append((role, content))

    def update_workspace_async(self) -> None:
        self.workspace_updates += 1


def test_stream_completion_parses_chunks_and_appends_assistant_message() -> None:
    chunks = [
        "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "thinking"}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}], "usage": {"completion_tokens": 2}}),
        "data: [DONE]",
    ]
    model = _FakeModel(_FakeResponse(chunks))

    usage = asyncio.run(stream_completion(model, max_completion_tokens=42))

    assert usage == {"completion_tokens": 2}
    assert model.append_calls == [("assistant", "Hello world")]
    # Workspace is refreshed once, after the turn completes.
    assert model.workspace_updates == 1
