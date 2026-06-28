"""Organization profile: the doc that describes the org and the rules governing
what gets extracted into shared org memory.

``load_org_profile`` accepts a path to a markdown doc, a literal profile string,
or ``None`` (falls back to the shipped template). The text is injected into the
extraction prompt so the model decides — per the rules — which facts are generic
and org-wide versus personal to one agent.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_PATH = Path(__file__).with_name("default_profile.md")


def load_org_profile(profile: str | Path | None) -> str:
    """Resolve an org profile to text.

    - ``None`` -> the shipped default template.
    - an existing file path -> its contents.
    - any other string -> treated as the profile text itself.
    """
    if profile is None:
        return _DEFAULT_PATH.read_text(encoding="utf-8")
    if isinstance(profile, Path):
        return profile.read_text(encoding="utf-8")
    candidate = Path(profile).expanduser()
    if "\n" not in profile and candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return profile


__all__ = ["load_org_profile"]
