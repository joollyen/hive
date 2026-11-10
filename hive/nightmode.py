"""Night Mode — the terminal becomes the engine, the voice keeps answering.

Between 22:00 and 06:00 speaking out loud is often not an option, but hearing an
answer still is. So the input channel flips: typed requests go through the exact
same pipeline a spoken one would — same wake-free routing, same intents, same
permissions, same spoken reply — and HIVE narrates each step as it works, because
at night that narration is the only signal that anything is happening.

Nothing is taken away. The microphone stays live by default; typing is preferred,
not enforced. The mode is a change of emphasis and presentation, not of
capability, which is why it can flip mid-session without disturbing anything.

The window wraps midnight, so the containment test is not a simple range check.
"""

from __future__ import annotations

import threading
from datetime import datetime

from .events import NIGHT_MODE, EventBus


def in_window(hour: int, start: int, end: int) -> bool:
    """Is `hour` inside a window that may wrap past midnight?

    22->6 is not `start <= hour < end`; that comparison is false for every hour.
    """
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


class NightMode:
    def __init__(self, cfg, bus: EventBus, runtime=None):
        self.cfg, self.bus = cfg, bus
        self.runtime = runtime
        self.enabled = bool(cfg.get("night.enabled", True))
        self.start = int(cfg.get("night.start_hour", 22))
        self.end = int(cfg.get("night.end_hour", 6))
        self._active = False
        self._forced: bool | None = None      # manual override beats the clock
        self._timer: threading.Timer | None = None
        self._stop = threading.Event()

    # -- state --------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    def scheduled_now(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        return in_window((now or datetime.now()).hour, self.start, self.end)

    def should_be_active(self) -> bool:
        return self._forced if self._forced is not None else self.scheduled_now()

    # -- lifecycle ----------------------------------------------------------
    def start_watching(self) -> None:
        self._evaluate(announce=False)
        self._schedule()

    def stop(self) -> None:
        self._stop.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule(self) -> None:
        if self._stop.is_set():
            return
        interval = float(self.cfg.get("night.check_interval_s", 60))
        self._timer = threading.Timer(interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        self._evaluate(announce=True)
        self._schedule()

    def _evaluate(self, announce: bool) -> None:
        want = self.should_be_active()
        if want == self._active:
            return
        self._active = want
        self.bus.publish(NIGHT_MODE, active=want)
        self.bus.log(f"night mode {'on' if want else 'off'}", "boot")
        if want and bool(self.cfg.get("night.auto_open_terminal", True)):
            from .events import TERMINAL_TOGGLE
            self.bus.publish(TERMINAL_TOGGLE, open=True)
        if announce and bool(self.cfg.get("night.announce_entry", True)) \
                and self.runtime is not None:
            bank = "night.entering" if want else "night.leaving"
            try:
                self.runtime.speaker.say_line(bank)
            except Exception:
                pass

    # -- manual control -----------------------------------------------------
    def set_forced(self, value: bool | None) -> str:
        """Override the schedule. None returns control to the clock."""
        self._forced = value
        self._evaluate(announce=False)
        if value is None:
            return ("Back on the schedule. Night mode runs "
                    f"{self.start}:00 to {self.end:02d}:00.")
        return "Night mode on. Terminal's the engine." if value else "Night mode off."

    def status_line(self) -> str:
        mode = "on" if self._active else "off"
        how = ("forced" if self._forced is not None
               else f"scheduled {self.start}:00-{self.end:02d}:00")
        return f"night mode {mode} ({how})"
