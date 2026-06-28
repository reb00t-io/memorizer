# Organization profile

> Replace this template with a real description of YOUR organization. The model
> reads this verbatim to decide what belongs in shared org memory, so be concrete
> and explicit — the org must be described obviously.

## Organization

- Name: Example Org
- What it does: (describe the company / team and its domain in 1–3 sentences)
- Audience of the shared memory: every agent and teammate in the organization.

## What org memory is for

Org memory holds **generic, durable, organization-wide knowledge** that is useful
to *any* agent or person in the org — not to one user or one conversation. It is
read-mostly, slow-moving, and shared across the whole fleet. Keep it small and
high-signal; it has **limited overlap** with an individual agent's personal memory.

## Extract into org memory (only if ALL apply)

- Generally applicable across the organization (not specific to one user/session).
- Durable and stable (policies, conventions, glossary terms, system facts,
  reusable lessons) — not a transient detail of the current task.
- Safe to share org-wide (no personal data, secrets, or private credentials).
- Not already covered by existing org knowledge (avoid duplicates / near-duplicates).

Examples to extract:
- "Deployments to production require two approvals."
- "The term 'tenant' refers to a customer workspace, not a server."
- "Internal tools are only reachable over the company VPN."

## Do NOT extract into org memory

- Anything specific to a single user, their preferences, or one conversation.
- Personal data, secrets, passwords, API keys, or private credentials.
- Transient state, one-off task details, or speculation.
- Anything you are not confident is true and organization-wide.

When in doubt, leave it out — it stays in the agent's personal memory instead.

## Roles and visibility

Members have roles (configured by the deployment). Roles gate org memory two ways:

- **Write**: only certain roles may promote facts into org memory. A member whose
  role lacks write access contributes nothing to org memory.
- **Read**: each org fact records which roles may see it. Most facts are visible to
  everyone (`"all"`); restrict a fact only when the rules above call for it.

When extracting, set each fact's `visible_to` to the roles that should see it, or
`"all"` if everyone may. Example: an HR-only policy → `["manager", "admin"]`; a
general engineering convention → `"all"`. Default to `"all"` unless a fact is
clearly sensitive to a subset of roles. Replace the example roles below with your
organization's real roles.

- Example roles: `engineer`, `manager`, `admin`.
