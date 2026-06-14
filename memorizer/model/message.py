from dataclasses import dataclass


@dataclass(slots=True)
class Message:
    role: str
    content: str
    compressed_content: str | None = None
    timestamp: str | None = None
    formatted_timestamp: str | None = None
