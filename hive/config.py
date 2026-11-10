"""Configuration loading for HIVE.

Three files, three jobs:
  config/hive.toml    — engineering knobs (latency, models, permissions)
  config/persona.yaml — everything HIVE says and how it says it
  config/apps.yaml    — what "open X" resolves to

Kept as plain dicts with dotted lookup rather than a dataclass tree. The config is
read on the hot path (every turn touches permissions and persona banks) and a
dotted string lookup against a dict is both faster to write and faster to run than
maintaining a parallel schema that has to be edited twice for every new knob.
"""

from __future__ import annotations

import random
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


class ShuffleBag:
    """Draws without replacement, reshuffling only when empty.

    Plain random.choice repeats roughly 1 in N times, and hearing the same
    "Right here." twice running is the single clearest tell that you are talking
    to a lookup table. A shuffle bag makes an immediate repeat impossible and a
    near repeat rare.
    """

    def __init__(self, items: list[str]):
        self._items = list(items) or [""]
        self._bag: list[str] = []
        self._last: str | None = None

    def draw(self) -> str:
        if not self._bag:
            self._bag = list(self._items)
            random.shuffle(self._bag)
            # Avoid the seam where a reshuffle puts the same line back-to-back.
            if len(self._bag) > 1 and self._bag[-1] == self._last:
                self._bag.insert(0, self._bag.pop())
        self._last = self._bag.pop()
        return self._last


class Config:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else ROOT
        cfgdir = self.root / "config"
        with (cfgdir / "hive.toml").open("rb") as fh:
            self.data: dict[str, Any] = tomllib.load(fh)
        self.persona: dict[str, Any] = yaml.safe_load(
            (cfgdir / "persona.yaml").read_text(encoding="utf-8")
        )
        apps_path = cfgdir / "apps.yaml"
        self.apps: dict[str, Any] = (
            yaml.safe_load(apps_path.read_text(encoding="utf-8")) if apps_path.exists() else {}
        )
        # Calibration profiles — VOSS and HackBot as lenses. Optional: a missing
        # file simply means HIVE is never calibrated, which is a valid setup.
        calib_path = cfgdir / "calibrations.yaml"
        self.calibrations: dict[str, Any] = (
            yaml.safe_load(calib_path.read_text(encoding="utf-8"))
            if calib_path.exists() else {}
        )
        self._bags: dict[str, ShuffleBag] = {}

    # -- dotted access ------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    # -- persona ------------------------------------------------------------
    def bank(self, path: str) -> list[str]:
        """Fetch a persona list by dotted path, e.g. 'ack.opening'."""
        node: Any = self.persona
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return []
            node = node[part]
        return list(node) if isinstance(node, list) else []

    def text(self, path: str, default: str = "") -> str:
        """One persona string, as opposed to a bank of alternatives.

        `bank` and `line` both assume a list, because almost everything HIVE says
        has several phrasings and varying them is the point. A few things are a
        single fixed wording on purpose — the status briefing is one, where the
        words are Jay's and the audio cache is keyed on the finished string — and
        those need a lookup that does not draw.
        """
        node: Any = self.persona
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node if isinstance(node, str) else default

    def line(self, path: str) -> str:
        """Draw one line from a persona bank, without immediate repeats."""
        if path not in self._bags:
            self._bags[path] = ShuffleBag(self.bank(path))
        return self._bags[path].draw()

    def timeslot(self, now: datetime | None = None) -> str:
        """Which time-of-day flavour applies right now."""
        h = (now or datetime.now()).hour
        if h < 5:
            return "late_night"
        if h < 9:
            return "early_morning"
        if h < 18:
            return "work_hours"
        return "evening"

    def presence_line(self, repeat: bool = False, now: datetime | None = None) -> str:
        """The 'HIVE, are you there?' response, time- and repetition-aware."""
        if repeat and self.bank("presence.repeat"):
            return self.line("presence.repeat")
        slot = self.timeslot(now)
        if self.bank(f"presence.{slot}"):
            # Mostly time-flavoured, but keep the default bank in rotation so it
            # never becomes predictable inside a single sitting.
            return self.line(f"presence.{slot}" if random.random() < 0.65 else "presence.default")
        return self.line("presence.default")

    # -- paths --------------------------------------------------------------
    @property
    def vault(self) -> Path:
        return Path(self.get("memory.vault_path", "~/vault")).expanduser()

    @property
    def hive_dir(self) -> Path:
        return self.vault / self.get("memory.hive_folder", "HIVE")

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def cache_dir(self) -> Path:
        d = self.root / "assets" / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d


_config: Config | None = None


def load(root: Path | None = None) -> Config:
    global _config
    if _config is None:
        _config = Config(root)
    return _config
