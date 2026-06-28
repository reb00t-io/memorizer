"""Role policy for shared organization memory.

Roles gate org memory two ways:

- **write**: only members whose role is in ``writer_roles`` may promote facts into
  org memory (an empty ``writer_roles`` means "no restriction").
- **read**: each org fact stores a ``visible_to`` role list; a member sees a fact
  only if their role is listed, or the fact is visible to everyone (``"*"``).

The sentinel ``"*"`` in ``visible_to`` means visible to all roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EVERYONE = "*"


@dataclass(frozen=True)
class OrgPolicy:
    roles: frozenset[str] = field(default_factory=frozenset)
    writer_roles: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def create(
        cls,
        *,
        roles: "set[str] | list[str] | None" = None,
        writer_roles: "set[str] | list[str] | None" = None,
    ) -> "OrgPolicy":
        return cls(
            roles=frozenset(roles or ()),
            writer_roles=frozenset(writer_roles or ()),
        )

    def can_write(self, role: str | None) -> bool:
        """True if a member with ``role`` may promote facts into org memory."""
        if not self.writer_roles:
            return True
        return role is not None and role in self.writer_roles

    def normalize_visibility(self, visible_to: object) -> list[str]:
        """Coerce an extraction-supplied audience into a stored ``visible_to`` list.

        ``None`` / empty / ``"all"`` / ``"*"`` -> visible to everyone. Otherwise
        keep only known roles (when ``roles`` is configured); if nothing valid
        remains, fall back to everyone rather than hiding the fact from all.
        """
        if visible_to is None:
            return [EVERYONE]
        if isinstance(visible_to, str):
            items = [visible_to]
        elif isinstance(visible_to, (list, tuple, set)):
            items = [str(v) for v in visible_to]
        else:
            return [EVERYONE]

        cleaned: list[str] = []
        for item in items:
            item = item.strip()
            if not item or item.lower() in ("all", "everyone", EVERYONE):
                return [EVERYONE]
            if self.roles and item not in self.roles:
                continue
            cleaned.append(item)
        return cleaned or [EVERYONE]


__all__ = ["OrgPolicy", "EVERYONE"]
