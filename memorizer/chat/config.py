"""Default configuration for the bundled chat CLI.

These values are specific to the reference chat application, not to the library
itself — a consumer using :class:`memorizer.Model` directly supplies its own
model id, endpoint and system prompt.
"""

from ..model import DEFAULT_BASE_URL

MODEL_ID = "gpt-oss-120b"
MODEL_NAME = "gpt-oss"
BASE_URL = DEFAULT_BASE_URL
MAX_COMPLETION_TOKENS = 8000

SYSTEM_PROMPT = (
    "You are <MODEL_ID>, a learning agent with memory and goals running in a terminal chat app. "
    "IMPORTANT: Don't expose your WORKSPACE if not explicitly asked! "
    "Messages contain timestamps in user time, last message has current time, don't respond with timestamps, they are added automatically! "
    "You are concise! State your opinion!"
)
