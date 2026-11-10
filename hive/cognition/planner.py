"""Multi-step request decomposition with tracked state.

τ-Voice (Sierra/Princeton, 2026) measured the failure this exists to prevent:
agents "lose track of multi-step requests, completing part of a task but
forgetting remaining items." Prompting does not fix that — the model has no
durable record of what it has already done, so under any pressure it drops a
step and reports success anyway.

So steps live in explicit state here, not in the model's context. Each carries
its own status, each is executed through the same objective hemisphere a single
request would use, and the closing report enumerates what happened to every one
of them — including the ones that failed or could not be done. A plan that
half-completes says so.

Splitting is deterministic and conservative. A request is only broken apart when
each fragment independently looks like something HIVE could be asked to do; a
sentence that merely contains "and" is left alone. Over-splitting is worse than
not splitting, because it turns one clear request into two confused ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Verbs that begin an actionable request. A fragment that does not start with one
# is prose, not a step, and splitting on it would be a mistake.
_ACTION_VERBS = (
    r"open|launch|start|fire up|bring up|pull up|show|go to|take me|"
    r"play|pause|resume|skip|stop|"
    r"make|create|write|draft|add|put|note|jot|save|record|remember|"
    r"check|search|find|look|tell|give|read|inspect|describe|"
    r"organi[sz]e|arrange|mirror|move|sort|"
    r"close|quit|switch|change|set"
)
_STARTS_ACTION = re.compile(rf"^\s*(?:{_ACTION_VERBS})\b", re.I)

# Boundaries strong enough to mean "separate request", strongest first.
_SPLITS = [
    re.compile(r"\s*(?:,|;)?\s*\band then\b\s*", re.I),
    re.compile(r"\s*(?:,|;)?\s*\bafter that\b\s*,?\s*", re.I),
    re.compile(r"\s*(?:,|;)?\s*\bthen\b\s+(?=(?:%s)\b)" % _ACTION_VERBS, re.I),
    re.compile(r"\s*(?:,|;)\s*\balso\b\s*,?\s*", re.I),
    re.compile(r"\s*(?:,|;)\s*\band also\b\s*", re.I),
    re.compile(r"\s*;\s*", re.I),
    re.compile(r"\s*,?\s*\band\b\s+(?=(?:%s)\b)" % _ACTION_VERBS, re.I),
]

# "first ... second ..." / "1) ... 2) ..."
_ENUMERATED = re.compile(
    r"(?:^|\s)(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|next|finally|lastly|"
    r"\(?\d\)|\d\.)\s+(?=(?:%s)\b)" % _ACTION_VERBS, re.I)

MAX_STEPS = 5


@dataclass
class Step:
    text: str                     # the sub-request, standing on its own
    index: int = 0
    status: str = "pending"       # pending | done | failed | refused | skipped
    spoken: str = ""              # what HIVE said while doing it
    label: str = ""               # short description for the closing report

    @property
    def ok(self) -> bool:
        return self.status == "done"


@dataclass
class Plan:
    origin: str
    steps: list[Step] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def done(self) -> list[Step]:
        return [s for s in self.steps if s.status == "done"]

    @property
    def unfinished(self) -> list[Step]:
        return [s for s in self.steps if s.status in ("failed", "refused", "skipped")]

    def opening_line(self) -> str:
        n = len(self.steps)
        word = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(n, str(n))
        return f"{word} things."

    def closing_line(self) -> str:
        """Enumerate the outcome of every step, including the ones that failed.

        This is the whole point of tracking state: a plan that half-completed
        must say so, in the same breath as what it did finish.
        """
        done, bad = self.done, self.unfinished
        if not bad:
            return "Both done." if len(done) == 2 else f"All {len(done)} done."
        did = ", ".join(s.label or s.text[:32] for s in done)
        failed = ", ".join(f"{s.label or s.text[:32]} ({s.status})" for s in bad)
        if done:
            return f"Done: {did}. Not done: {failed}."
        return f"None of that worked. {failed}."


def split_steps(text: str) -> list[str]:
    """Break a request into independent sub-requests, or return one item."""
    text = (text or "").strip()
    if not text:
        return []

    parts = _ENUMERATED.split(text) if len(_ENUMERATED.findall(text)) >= 1 else [text]
    for pattern in _SPLITS:
        nxt: list[str] = []
        for part in parts:
            nxt.extend(pattern.split(part))
        parts = nxt

    cleaned = [p.strip(" ,.;") for p in parts if p and p.strip(" ,.;")]
    if len(cleaned) < 2:
        return [text]

    # Every fragment after the first must look actionable on its own. If any does
    # not, the split was grammatical rather than semantic — leave it whole.
    if not all(_STARTS_ACTION.match(p) for p in cleaned):
        return [text]
    if any(len(p) < 4 for p in cleaned):
        return [text]
    return cleaned[:MAX_STEPS]


def make_plan(text: str) -> Plan | None:
    """A Plan when the request is genuinely multi-step, otherwise None."""
    parts = split_steps(text)
    if len(parts) < 2:
        return None
    return Plan(origin=text,
                steps=[Step(text=p, index=i) for i, p in enumerate(parts)])
