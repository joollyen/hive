"""HIVE's state machine and event bus.

The bus is deliberately synchronous and lock-guarded rather than asyncio-based.
Publishers live on four different threads — the PortAudio capture callback, the
VAD/segmentation thread, the asyncio engine thread, and the Qt main thread — and
a plain lock is the only thing all four can use without a loop reference. The
audio callback in particular must never touch an event loop; missing a deadline
there produces an audible click.

Subscribers must not block. Anything slow re-posts to its own queue.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class State(str, Enum):
    """What HIVE is doing. Drives both behaviour and the visualizer."""

    DORMANT = "dormant"          # awake but not addressed; wake word required
    LISTENING = "listening"      # capturing speech, columns track the mic
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"        # LLM working; status word shown
    SPEAKING = "speaking"        # columns track TTS output
    CONFIRMING = "confirming"    # awaiting a spoken yes/no
    ALERT = "alert"              # veto or failure; red accent
    CLOSING = "closing"          # consolidating to the vault


# Event names. Strings rather than an enum so the QML side can use them directly.
STATE_CHANGED = "state.changed"
AUDIO_LEVEL = "audio.level"          # {rms, peak, spectrum: list[float]}
TRANSCRIPT_PARTIAL = "transcript.partial"
TRANSCRIPT_FINAL = "transcript.final"
SPEAK_START = "speak.start"
SPEAK_END = "speak.end"
SPEAK_CHUNK = "speak.chunk"          # {text, first} one synthesis unit, as it is queued
STATUS_WORD = "status.word"          # {word} e.g. "Cross-referencing"
LOG_LINE = "log.line"                # feeds the HUD glyph stream
METRIC = "metric"                    # {name, ms} latency instrumentation
ENGINE_READY = "engine.ready"        # models warm; UI lifts the boot veil
TERMINAL_TOGGLE = "terminal.toggle"  # {open: bool} show/hide the HIVE terminal
TERMINAL_OUTPUT = "terminal.output"  # {text, kind} a line for the terminal panel
CLIPBOARD_SET = "clipboard.set"      # {text} — only the UI thread can touch QClipboard
NIGHT_MODE = "night.mode"            # {active: bool}
CALIBRATION = "calibration.changed"  # {key, label, previous} silent lens change
PLAN_PROGRESS = "plan.progress"      # {index, total, label, status}
SESSION_END = "session.end"
# {stage, name, path, verified} — consolidation reporting on itself. Published
# twice on a normal shutdown: once when the transcript is safe on disk, once
# when the summary has been written over the top of it. The closing veil reads
# these, so "did it save" is answered on screen instead of in the vault folder.
SESSION_SAVED = "session.saved"

# -- HIVE-Q ------------------------------------------------------------------
# The query runs on its own thread for minutes at a time with nobody watching it,
# so every transition it makes is published rather than kept internal. The HIVE-Q
# window is built entirely from these three, and so is the runner's own watchdog:
# QUERY_* and the speech events together are what "the task is still making
# progress" is measured from.
QUERY_STATE = "query.state"          # {running, paused, index, total, stopping}
QUERY_TASK = "query.task"            # {index, total, status, text, reason, note, ...}
QUERY_OUTPUT = "query.output"        # {text, kind} a line for the HIVE-Q log pane
QUERY_WINDOW = "query.window"        # {open: bool} show/hide the HIVE-Q window
WEB_RESULTS = "web.results"          # {query, results: [{title, url, snippet}]}


@dataclass
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.monotonic)


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.RLock()
        self._state = State.DORMANT
        self._state_since = time.monotonic()

    # -- pub/sub ------------------------------------------------------------
    def subscribe(self, event: str, handler: Handler) -> None:
        with self._lock:
            self._subs[event].append(handler)

    # `event` is positional-only so a payload key called "name" — which metric()
    # and any future caller will naturally want — cannot collide with it.
    def publish(self, event: str, /, **payload: Any) -> None:
        evt = Event(event, payload)
        with self._lock:
            handlers = list(self._subs.get(event, ()))
            handlers += list(self._subs.get("*", ()))
        for h in handlers:
            try:
                h(evt)
            except Exception:
                # A broken subscriber must never take down the audio path.
                traceback.print_exc()

    # -- state --------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    @property
    def state_age(self) -> float:
        return time.monotonic() - self._state_since

    def set_state(self, state: State, **detail: Any) -> None:
        if state is self._state:
            return
        prev, self._state = self._state, state
        self._state_since = time.monotonic()
        self.publish(STATE_CHANGED, state=state.value, previous=prev.value, **detail)

    # -- convenience --------------------------------------------------------
    def log(self, text: str, kind: str = "info") -> None:
        """Feeds the scrolling glyph stream in the visor, and stderr in debug."""
        self.publish(LOG_LINE, text=text, kind=kind)

    def metric(self, name: str, ms: float) -> None:
        self.publish(METRIC, name=name, ms=ms)


class Stopwatch:
    """Marks named stages of a turn so the latency budget is measurable.

    Every turn produces one of these. If first-audio overruns the budget the whole
    breakdown gets logged, which is the only practical way to find out whether the
    time went to VAD hangover, ASR, the model, or the vocoder.
    """

    def __init__(self, bus: EventBus | None = None, label: str = "turn"):
        self.bus = bus
        self.label = label
        self.t0 = time.monotonic()
        self.marks: list[tuple[str, float]] = []

    def mark(self, name: str) -> float:
        ms = (time.monotonic() - self.t0) * 1000.0
        self.marks.append((name, ms))
        if self.bus:
            self.bus.metric(f"{self.label}.{name}", ms)
        return ms

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.t0) * 1000.0

    def breakdown(self) -> str:
        parts, prev = [], 0.0
        for name, ms in self.marks:
            parts.append(f"{name}+{ms - prev:.0f}")
            prev = ms
        return " ".join(parts) + f" = {prev:.0f}ms"
