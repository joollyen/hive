"""Calibration — VOSS and HackBot as lenses HIVE adopts, not agents it hands off to.

Jay has two other agents he has spent a long time tuning: HackBot, an offensive
security and systems co-driver, and VOSS, a meta-agent for architecture and
self-analysis. Both are better than HIVE at their own domain, and neither should
become a *destination*. The brief was explicit: no asking, no announcement, no
handoff ceremony. HIVE is the agent. These are settings.

So a calibration is applied for exactly one turn and dropped. What it changes is
the operating spine, the grounded facts, the length budget, whether the heavy
model is used, and which bank a follow-up question is drawn from. What it never
changes is the voice, the permission model, or the fact that HIVE is speaking —
a profile written for a text chat will happily emit a code block into a
text-to-speech engine, so HIVE's own spine is always prepended, never replaced.

Three design decisions worth keeping:

**Routing is lexical, not learned.** A model call to decide which model to call
would cost more than the switch is worth, and the whole point is that it is
instantaneous. Scored keyword matching over a config file resolves in tens of
microseconds and — more importantly — is auditable: `calibration why <text>` in
the terminal prints exactly why a turn went where it went.

**Actions are never routed.** Calibration is only ever considered on the path
that reaches the language model. "Open Spotify", "check Indeed for overnight
roles", "note that down" all match a concrete intent and never get here, which
is what makes the rule "simple tasks stay HIVE" structural rather than a
heuristic that might one day misfire on the wrong keyword.

**A thread holds; a turn does not.** Strict per-turn selection drops calibration
on the first follow-up that happens not to repeat a keyword — "and what about
the hash?" would fall back to HIVE mid-thread. So a calibration persists for a
couple of follow-up turns and is broken immediately by any action intent, which
keeps "default back to HIVE" true without severing a conversation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..events import CALIBRATION, EventBus

BASE = "hive"


@dataclass
class Calibration:
    key: str
    label: str
    spine: str = ""
    facts: dict[str, str] = field(default_factory=dict)
    budget: str = ""
    escalate: bool = False
    probes: str = ""

    @property
    def is_base(self) -> bool:
        return self.key == BASE


HIVE = Calibration(BASE, "HIVE")


def _pattern(terms: list[str]) -> re.Pattern | None:
    """One alternation over a term list, word-bounded at both ends.

    Bounded on both sides so `hash` does not fire on `hashtag` and `port` does
    not fire on `portfolio` — the second of which would have routed every resume
    question to HackBot.
    """
    terms = [t.strip().lower() for t in terms if t and t.strip()]
    if not terms:
        return None
    # Longest first so the alternation prefers the most specific match.
    terms.sort(key=len, reverse=True)
    body = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.I)


@dataclass
class Match:
    key: str
    score: int
    hits: list[str] = field(default_factory=list)


class Calibrator:
    def __init__(self, cfg, bus: EventBus):
        self.cfg, self.bus = cfg, bus
        self.enabled = bool(cfg.get("calibration.enabled", True))
        self._min_score = int(cfg.get("calibration.min_score", 2))
        self._hold_turns = int(cfg.get("calibration.hold_turns", 2))

        self.profiles: dict[str, Calibration] = {BASE: HIVE}
        self._routes: dict[str, tuple[re.Pattern | None, re.Pattern | None]] = {}
        self._priority: dict[str, int] = {}
        self._load()

        self.current: Calibration = HIVE
        self._held = 0
        # Wall-clock and turn counts per calibration, for the session note. The
        # active calibration owns the clock until a different one takes it, so
        # thinking time and silence land on whichever lens was last in use —
        # which is how the operator describes a session ("45 minutes with VOSS").
        self.elapsed: dict[str, float] = {}
        self.turns: dict[str, int] = {}
        self._since = time.monotonic()
        self._last_reason = ""

    # -- config -------------------------------------------------------------
    def _load(self) -> None:
        data = getattr(self.cfg, "calibrations", {}) or {}
        for key, spec in data.items():
            if not isinstance(spec, dict):
                continue
            self.profiles[key] = Calibration(
                key=key,
                label=str(spec.get("label", key.upper())),
                spine=str(spec.get("spine", "")).strip(),
                facts={k: " ".join(str(v).split())
                       for k, v in (spec.get("facts") or {}).items()},
                budget=str(spec.get("budget", "")).strip(),
                escalate=bool(spec.get("escalate", True)),
                probes=str(spec.get("probes", "")).strip(),
            )
            route = spec.get("route") or {}
            self._routes[key] = (_pattern(route.get("strong") or []),
                                 _pattern(route.get("weak") or []))
            # File order is the tiebreak, and it is load-bearing rather than
            # incidental: "explain the architecture of Active Directory" scores
            # two for each profile, and it is a HackBot question wearing an
            # architectural word. The more concrete domain wins a tie, so
            # HackBot is listed first in the yaml.
            self._priority[key] = len(self._priority)
        if len(self.profiles) > 1:
            self.bus.log(
                "calibrations loaded: "
                + ", ".join(p.label for k, p in self.profiles.items() if k != BASE),
                "boot")

    # -- routing ------------------------------------------------------------
    def score(self, text: str) -> list[Match]:
        """Every calibration's score for this text, best first. Auditable."""
        out: list[Match] = []
        for key, (strong, weak) in self._routes.items():
            hits: list[str] = []
            score = 0
            if strong is not None:
                found = {m.group(0).lower() for m in strong.finditer(text)}
                score += 2 * len(found)
                hits += sorted(found)
            if weak is not None:
                found = {m.group(0).lower() for m in weak.finditer(text)}
                score += len(found)
                hits += sorted(found)
            if score:
                out.append(Match(key, score, hits))
        out.sort(key=lambda m: (-m.score, self._priority.get(m.key, 99)))
        return out

    def select(self, intent, text: str) -> Calibration:
        """Pick the lens for this turn. Only ever called on the generative path."""
        if not self.enabled:
            return self._switch(HIVE)

        matches = self.score(text or "")
        best = matches[0] if matches else None

        if best is not None and best.score >= self._min_score:
            self._held = self._hold_turns
            self._last_reason = (f"{self.profiles[best.key].label} "
                                 f"score {best.score} on {', '.join(best.hits[:6])}")
            return self._switch(self.profiles[best.key])

        # Nothing matched. Hold a thread that is plainly still running rather
        # than dropping calibration on the first follow-up phrased as "and that?"
        if not self.current.is_base and self._held > 0:
            self._held -= 1
            self._last_reason = f"{self.current.label} held ({self._held} left)"
            return self.current

        self._last_reason = "no calibration matched"
        return self._switch(HIVE)

    def release(self) -> None:
        """Drop calibration immediately — an action intent ends the thread."""
        self._held = 0
        if not self.current.is_base:
            self._switch(HIVE)

    # -- accounting ---------------------------------------------------------
    def _switch(self, to: Calibration) -> Calibration:
        if to.key == self.current.key:
            return to
        now = time.monotonic()
        self.elapsed[self.current.key] = (
            self.elapsed.get(self.current.key, 0.0) + (now - self._since))
        self._since = now
        previous, self.current = self.current, to
        # The only notification anywhere. Not spoken, not on the visualizer —
        # one line in the terminal, which is all the operator asked to see.
        self.bus.publish(CALIBRATION, key=to.key, label=to.label,
                         previous=previous.label)
        return to

    def note_turn(self) -> None:
        self.turns[self.current.key] = self.turns.get(self.current.key, 0) + 1

    def close(self) -> None:
        """Settle the clock so the session note's totals add up."""
        now = time.monotonic()
        self.elapsed[self.current.key] = (
            self.elapsed.get(self.current.key, 0.0) + (now - self._since))
        self._since = now

    def stats(self) -> list[tuple[str, float, int]]:
        """(label, minutes, turns) per calibration, busiest first."""
        self.close()
        rows = []
        for key in set(self.elapsed) | set(self.turns):
            label = self.profiles.get(key, HIVE).label
            rows.append((label, self.elapsed.get(key, 0.0) / 60.0,
                         self.turns.get(key, 0)))
        rows.sort(key=lambda r: -r[1])
        return rows

    def summary(self) -> str:
        """Human-readable state, for the terminal's `calibration` command."""
        lines = [f"active     {self.current.label}",
                 f"last call  {self._last_reason or '—'}"]
        rows = self.stats()
        if rows:
            lines.append("")
            lines.append("  lens      time     turns")
            for label, mins, turns in rows:
                lines.append(f"  {label:<9s} {mins:5.1f}m   {turns:3d}")
        return "\n".join(lines)

    def explain(self, text: str) -> str:
        """Why a given sentence would route where it does. The audit trail."""
        matches = self.score(text)
        if not matches:
            return f"'{text[:60]}' -> HIVE (nothing matched)"
        lines = []
        for m in matches:
            verdict = "engages" if m.score >= self._min_score else "below threshold"
            lines.append(f"  {self.profiles[m.key].label:<9s} {m.score:2d}  "
                         f"{verdict}   [{', '.join(m.hits[:8])}]")
        winner = matches[0]
        head = (self.profiles[winner.key].label
                if winner.score >= self._min_score else "HIVE")
        return f"'{text[:60]}' -> {head}\n" + "\n".join(lines)
