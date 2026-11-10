"""Short-term memory — the working set for the current session.

Lives entirely in RAM and dies with the process, which is the point: HIVE is
session-scoped, so short-term memory is session-scoped too. Anything worth
surviving gets consolidated into the vault on close.

Two things happen here that a plain transcript list would not do:

1. Compaction. Verbatim history is kept for the last N turns; everything older is
   folded into a running summary. Prompt length is the main driver of time-to-
   first-token, so letting history grow without bound would make HIVE get
   measurably slower the longer you talk to it — the exact opposite of what a
   session-scoped assistant should feel like.

2. Salience. Terms that keep coming up are tracked with recency-weighted scores.
   That gives "what did we say about the resume" something concrete to anchor to,
   and it decides what is worth writing to the vault at the end.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str            # "user" | "hive"
    text: str
    intent: str = ""
    ts: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    # Which lens answered — "hive", "voss", "hackbot". Recorded per turn rather
    # than per session because a single session moves between all three, and the
    # session note has to be able to say which one said what.
    calibration: str = "hive"


# Technical vocabulary and proper nouns carry the meaning in this domain; the
# stopword list keeps ordinary sentence scaffolding out of the salience map.
_TERM = re.compile(r"\b([A-Z][a-zA-Z0-9+#.-]{2,}|[a-z]+[A-Z][a-zA-Z]+|\b[a-z]{4,}\b)")
_STOP = {
    "the", "and", "that", "this", "with", "from", "have", "what", "when", "your",
    "about", "would", "there", "their", "which", "could", "should", "these",
    "those", "into", "just", "like", "want", "need", "make", "know", "think",
    "going", "really", "something", "anything", "everything", "hive", "okay",
    "yeah", "right", "good", "well", "sure", "thing", "things", "some", "more",
    "much", "very", "then", "than", "them", "they", "been", "were", "will",
    # Added in v5. Salience is read by three things that all present it to a
    # human — the curiosity probe, the session note's term list, and the
    # write-up's — and a list reading "prosody · matters · most · here · used"
    # makes all three look like a word-frequency dump. None of these can be the
    # subject of a sentence about the session, however often they appear in one.
    "here", "most", "many", "each", "every", "such", "both", "also", "only",
    "even", "still", "always", "never", "often", "again", "already", "almost",
    "used", "using", "happens", "happen", "read", "word", "words", "matters",
    "matter", "distance", "different", "first", "second", "third", "next",
    "last", "other", "another", "three", "four", "five", "take", "takes",
    "made", "come", "comes", "came", "does", "done", "said", "says", "tell",
    "tells", "told", "keep", "keeps", "kept", "look", "looks", "find", "finds",
    "give", "gives", "gets", "getting", "part", "parts", "kind", "kinds",
    "sort", "sorts", "ways", "time", "times", "lots", "bits", "sense", "point",
    "points", "case", "cases", "level", "levels", "stuff", "whole", "actually",
    "basically", "probably", "definitely", "maybe", "perhaps", "instead",
    "where", "whether", "though", "however", "therefore", "through", "without",
    "within", "between", "against", "across", "around", "above", "below",
}


class ShortTermMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_turns = int(cfg.get("memory.stm_max_turns", 40))
        self.summarize_after = int(cfg.get("memory.stm_summary_after", 24))
        self.turns: list[Turn] = []
        self.rolling_summary: str = ""
        self._salience: dict[str, float] = defaultdict(float)
        self.started = time.time()
        self.pending_confirm: dict | None = None   # action awaiting a spoken yes/no
        self.last_presence_ts: float = 0.0

    # -- recording ----------------------------------------------------------
    def add_user(self, text: str, intent: str = "") -> None:
        self.turns.append(Turn("user", text, intent))
        self._score(text, weight=1.0)
        self._trim()

    def add_hive(self, text: str, latency_ms: float = 0.0,
                 calibration: str = "hive") -> None:
        self.turns.append(Turn("hive", text, latency_ms=latency_ms,
                               calibration=calibration))
        self._score(text, weight=0.35)   # HIVE's own words matter less to salience
        self._trim()

    def _score(self, text: str, weight: float) -> None:
        # Decay everything slightly each turn so salience tracks recency, not just
        # raw frequency. Without this, whatever was discussed first dominates the
        # entire session.
        for k in list(self._salience):
            self._salience[k] *= 0.97
        for m in _TERM.finditer(text):
            term = m.group(1)
            low = term.lower()
            if low in _STOP or len(low) < 4:
                continue
            bonus = 1.5 if term[0].isupper() or any(c.isupper() for c in term[1:]) else 1.0
            self._salience[low] += weight * bonus

    def _trim(self) -> None:
        if len(self.turns) <= self.max_turns:
            return
        overflow = self.turns[: len(self.turns) - self.summarize_after]
        self.turns = self.turns[len(self.turns) - self.summarize_after:]
        lines = [f"{t.role}: {t.text}" for t in overflow if t.text]
        if lines:
            merged = " | ".join(lines)
            self.rolling_summary = (self.rolling_summary + " | " + merged)[-2400:]

    # -- retrieval ----------------------------------------------------------
    def salient(self, n: int = 12) -> list[tuple[str, float]]:
        return sorted(self._salience.items(), key=lambda kv: -kv[1])[:n]

    def recent_text(self, n: int = 6) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.turns[-n:])

    def last_user_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "user":
                return t.text
        return ""

    def messages(self, system: str, extra_context: str = "") -> list[dict]:
        """Assemble the model prompt: system, compacted history, live turns."""
        msgs: list[dict] = [{"role": "system", "content": system}]
        if self.rolling_summary:
            msgs.append({
                "role": "system",
                "content": f"Earlier in this session: {self.rolling_summary}",
            })
        if extra_context:
            msgs.append({"role": "system", "content": extra_context})
        for t in self.turns:
            if not t.text:
                continue
            msgs.append({
                "role": "user" if t.role == "user" else "assistant",
                "content": t.text,
            })
        return msgs

    # -- session stats ------------------------------------------------------
    @property
    def duration(self) -> float:
        return time.time() - self.started

    @property
    def user_turns(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def transcript(self) -> list[Turn]:
        return list(self.turns)
