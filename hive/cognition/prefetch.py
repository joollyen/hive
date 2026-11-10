"""Slow Thinker — background work during the user's listening time.

Adapted from VoiceAgentRAG (Salesforce, 2026), which pairs a background agent
that predicts and prefetches with a foreground agent that reads cache-first. The
paper's target is vector-store round trips; HIVE's vault search is already
sub-millisecond, so that half does not transfer.

What does transfer is the **window**. While HIVE is speaking, the user is
listening — several seconds of compute nobody is waiting on. HIVE's worst
measured latency by a wide margin is the ~4 second cold load of the 8B model. If
we can predict during turn N that turn N+1 will want it, that load happens for
free and the user never sees it.

The prediction is deliberately cheap and heuristic rather than a second LLM call:
a wrong guess costs VRAM for the keep-alive window and nothing else, so an
expensive prediction would cost more than the mistake it prevents.

Also warms the vault: terms that are live in the conversation get their search
results cached, so a follow-up recall skips the filesystem walk.
"""

from __future__ import annotations

import threading
import time

from ..events import EventBus

# Categories where the next turn is very likely to want real reasoning.
_HEAVY_CATEGORIES = {"analysis", "synthesis", "writing", "research", "memory"}


class SlowThinker:
    def __init__(self, cfg, bus: EventBus, llm, ltm, stm):
        self.cfg, self.bus = cfg, bus
        self.llm, self.ltm, self.stm = llm, ltm, stm
        self.enabled = bool(cfg.get("prefetch.enabled", True))
        self._cooldown = float(cfg.get("prefetch.cooldown_s", 45))
        self._last_warm = 0.0
        self._lock = threading.Lock()
        self._vault_cache: dict[str, list] = {}
        self.hits = 0
        self.warms = 0

    # -- prediction ---------------------------------------------------------
    def predicts_heavy(self, category: str, escalated: bool, text: str) -> bool:
        """Will the next turn want the 8B model?

        Three signals, in order of how well they held up in practice:
          - this turn already escalated: conversations stay in register
          - the topic category is one that reasons rather than acts
          - the user is writing long utterances, which correlates with depth
        """
        if escalated:
            return True
        if category in _HEAVY_CATEGORIES:
            return True
        return len(text.split()) >= 14

    # -- background work ----------------------------------------------------
    def after_turn(self, category: str, escalated: bool, text: str) -> None:
        """Called once a turn's response is under way. Never blocks the caller."""
        if not self.enabled:
            return
        threading.Thread(target=self._work, args=(category, escalated, text),
                         name="hive-slowthinker", daemon=True).start()

    def _work(self, category: str, escalated: bool, text: str) -> None:
        try:
            if self.predicts_heavy(category, escalated, text):
                self._prewarm_heavy()
            self._prefetch_vault()
        except Exception as exc:                       # never disturb a live turn
            self.bus.log(f"prefetch failed: {exc}", "warn")

    def _prewarm_heavy(self) -> None:
        """Load the 8B into VRAM now, so the next turn does not pay for it.

        Rate-limited: repeatedly reloading a 5GB model would cost more than the
        cold start it is meant to hide.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_warm < self._cooldown:
                return
            self._last_warm = now
        if not self.llm.available():
            return
        t0 = time.monotonic()
        try:
            self.llm.complete(
                [{"role": "user", "content": "ok"}], heavy=True, num_predict=1)
            self.warms += 1
            self.bus.log(f"prewarmed heavy model ({(time.monotonic()-t0)*1000:.0f}ms)",
                         "llm")
        except Exception:
            pass

    def _prefetch_vault(self) -> None:
        """Cache vault hits for whatever is currently salient."""
        for term, score in self.stm.salient(3):
            if score < 1.5 or term in self._vault_cache:
                continue
            try:
                self._vault_cache[term] = self.ltm.search(term, limit=4)
            except Exception:
                continue
        # Bounded: this is a warm cache, not a second memory.
        if len(self._vault_cache) > 24:
            for key in list(self._vault_cache)[:12]:
                self._vault_cache.pop(key, None)

    def vault_lookup(self, query: str) -> list | None:
        """Cache-first vault read. Returns None on a miss so the caller falls back."""
        key = query.strip().lower()
        for term, hits in self._vault_cache.items():
            if term in key:
                self.hits += 1
                return hits
        return None

    def stats(self) -> str:
        return f"prefetch: {self.warms} warms, {self.hits} cache hits"
