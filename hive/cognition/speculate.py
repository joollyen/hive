"""Reading the draft as it is typed, and getting ahead of the answer.

The terminal is the primary channel at night, and typing has a property speech
does not: the request exists, partially, for several seconds before it is
submitted. Those seconds are free. HIVE spends them here.

What gets warmed, in the order it pays off:

  the holding line   The line HIVE will say if the answer runs late is composed
                     from the request and synthesized *before Enter is pressed*.
                     By the time it is needed it is a cache hit, so the tailored
                     acknowledgement — the one that proves the question was
                     understood — costs nothing and can fire in a quarter of a
                     second instead of two thirds.
  the heavy model    The worst measured latency in the system by a wide margin
                     is the ~4 second cold load of the 8B. If the half-typed
                     draft already looks like it will escalate, that load starts
                     while he is still typing and he never sees it.
  the vault          Association lookups for whatever is already on the line.

What deliberately does NOT get warmed: **the answer itself.**

Speculative generation is the obvious next step and it is the wrong one here.
Answering a half-written question means burning the VRAM that the real answer
needs, on a prompt the operator has not finished writing — and the failure mode
is not a wasted cycle, it is a confident answer to a question that was about to
say "...on Windows" and now says "...on Linux". The warm model and the ready
acknowledgement are where the perceived latency actually lives; the generation
itself waits for a complete request, as it should.

Everything here is best-effort and silent. A speculation that turns out to be
wrong costs one unused cache entry.
"""

from __future__ import annotations

import threading
import time

from .intents import normalize
from .phrasing import holding_line

MIN_CHARS = 14
MIN_WORDS = 4


class Speculator:
    def __init__(self, cfg, bus, speaker, llm, prefetch=None, calibrator=None):
        self.cfg, self.bus = cfg, bus
        self.speaker, self.llm = speaker, llm
        self.prefetch, self.calibrator = prefetch, calibrator
        self.enabled = bool(cfg.get("speculate.enabled", True))
        self._debounce = float(cfg.get("speculate.debounce_ms", 220)) / 1000.0
        self._min_gap = float(cfg.get("speculate.min_gap_ms", 800)) / 1000.0

        self._draft = ""
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._last_line = ""
        self._last_synth = 0.0
        self._worker: threading.Thread | None = None
        self.rendered = 0
        self.warmed = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        self._worker = threading.Thread(target=self._loop, name="hive-speculate",
                                        daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    # -- input --------------------------------------------------------------
    def draft(self, text: str) -> None:
        """Called on every keystroke, from the GUI thread. Must stay trivial."""
        if not self.enabled:
            return
        self._draft = text or ""
        self._wake.set()

    # -- worker -------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            if self._stop.is_set():
                return
            self._wake.clear()
            # Coalesce a burst of keystrokes into one pass: wait out the
            # debounce, and if more arrived in the meantime, wait again. Typing
            # at speed should produce work at the pauses, not per character.
            while True:
                time.sleep(self._debounce)
                if not self._wake.is_set():
                    break
                self._wake.clear()
            try:
                self._consider(self._draft)
            except Exception as exc:              # never disturb the terminal
                self.bus.log(f"speculation failed: {exc}", "warn")

    def _consider(self, raw: str) -> None:
        text = (raw or "").strip()
        if len(text) < MIN_CHARS or len(text.split()) < MIN_WORDS:
            return
        # A command is not a question. `grant`, `recall`, `tree` and friends are
        # handled without the model and have nothing to warm.
        head = text.split()[0].lower()
        if head in _COMMAND_HEADS:
            return

        cleaned = normalize(text)
        # Render first. Warming the heavy model is a multi-second blocking call
        # and the holding line is a hundred-millisecond one, so doing them in
        # the obvious order meant the line was still unrendered when Enter was
        # pressed — the exact thing this class exists to prevent. The warm now
        # runs on its own thread and the render never waits behind it.
        self._render_line(cleaned)
        self._warm_model(cleaned)
        self._warm_vault(cleaned)

    def _warm_model(self, cleaned: str) -> None:
        """Start the 8B load now if this draft is heading for it."""
        if self.llm is None or not self.prefetch:
            return
        wants_heavy = self.llm.should_escalate(cleaned)
        if not wants_heavy and self.calibrator is not None:
            # A calibrated turn always escalates, so a draft that already scores
            # for HackBot or VOSS is a draft that will want the heavy model.
            matches = self.calibrator.score(cleaned)
            wants_heavy = bool(matches and matches[0].score >= 2)
        if not wants_heavy:
            return

        def warm() -> None:
            before = self.prefetch.warms
            self.prefetch._prewarm_heavy()       # rate-limited internally
            if self.prefetch.warms > before:
                self.warmed += 1

        threading.Thread(target=warm, name="hive-spec-warm", daemon=True).start()

    def _render_line(self, cleaned: str) -> None:
        """Synthesize the holding line before it is asked for."""
        line = holding_line(self.cfg, cleaned)
        if not line or line == self._last_line:
            return
        now = time.monotonic()
        if now - self._last_synth < self._min_gap:
            return
        # Never contend with live speech for the GPU. The whole point is to be
        # invisible, and a speculative render that delays a real answer is not.
        if self.speaker.speaking or not self.speaker.ready:
            return
        self._last_line, self._last_synth = line, now
        self.speaker.synth(line, use_cache=True)
        self.rendered += 1

    def _warm_vault(self, cleaned: str) -> None:
        if self.prefetch is not None:
            try:
                self.prefetch._prefetch_vault()
            except Exception:
                pass

    def stats(self) -> str:
        return (f"speculation: {self.rendered} lines pre-rendered, "
                f"{self.warmed} model warms")


# Terminal verbs. Typing one of these is not asking a question.
_COMMAND_HEADS = {
    "grant", "revoke", "grants", "find", "tree", "structure", "mirror", "apply",
    "voices", "voice", "night", "jobs", "recall", "calibration", "calib",
    "clear", "help", "ls", "list",
}
