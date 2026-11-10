"""Launching applications, URLs and files.

Wayland notes, because they shape this whole module:

  - Launching is fine. Anyone can exec a process or hand a URL to a browser.
  - *Manipulating* existing windows is not. Wayland deliberately denies clients
    the ability to focus, move, resize or enumerate other applications' windows.
    There is no xdotool equivalent. So HIVE opens things; it does not rearrange
    them. Attempting otherwise would need a GNOME Shell extension, which is a
    different project.

Every launch is detached with start_new_session=True. Without it the child sits in
HIVE's process group and dies when HIVE closes — which would make "open Chrome"
followed by quitting HIVE close Chrome, an genuinely infuriating bug.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

DESKTOP_DIRS = [
    Path("/usr/share/applications"),
    Path.home() / ".local/share/applications",
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/flatpak/exports/share/applications",
    Path("/var/lib/snapd/desktop/applications"),
]


def _contains(haystack: str, needle: str) -> bool:
    """Is `needle` present in `haystack` as whole words?

    Plain substring matching is what let "mail" (an alias of Gmail) match inside
    "email the recruiter" and "docs" match inside "the docs on my desktop". The
    aliases are things people say out loud, so they are words, and a word
    boundary is the difference between recognising one and finding it by
    accident inside another.
    """
    if len(needle) < 4 or len(haystack) < 4:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


@dataclass
class Target:
    kind: str           # "app" | "url" | "file"
    label: str          # what HIVE calls it when confirming
    argv: list[str]
    score: float


class Launcher:
    def __init__(self, cfg, bus):
        self.cfg, self.bus = cfg, bus
        self.apps: dict = cfg.apps.get("apps", {}) or {}
        self.web: dict = cfg.apps.get("web", {}) or {}
        self.browsers: dict = cfg.apps.get("browsers", {}) or {}
        self._desktop_index: dict[str, Path] | None = None

    # -- desktop entry discovery -------------------------------------------
    def _index_desktop(self) -> dict[str, Path]:
        """Map display name -> .desktop path for everything installed.

        Built lazily and cached: this is the long-tail fallback for apps that are
        not in apps.yaml, so it only matters when an explicit alias misses.
        """
        if self._desktop_index is not None:
            return self._desktop_index
        index: dict[str, Path] = {}
        for d in DESKTOP_DIRS:
            if not d.is_dir():
                continue
            for f in d.glob("*.desktop"):
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if "NoDisplay=true" in text:
                    continue
                name = ""
                for line in text.splitlines():
                    if line.startswith("Name=") and not name:
                        name = line[5:].strip()
                        break
                if name:
                    index.setdefault(name.lower(), f)
        self._desktop_index = index
        return index

    # -- resolution ---------------------------------------------------------
    def resolve(self, spoken: str, want_window: bool = False,
                browser: str | None = None) -> Target | None:
        """Turn a spoken target into something executable.

        Order is deliberate: web shortcuts before apps, because "open a new doc"
        should create a document, not launch LibreOffice; and explicit aliases
        before the fuzzy desktop scan, because the scan will happily match
        "Claude" to "Chess - Play & Learn" given a bad enough transcript.
        """
        t = (spoken or "").strip().lower()
        if not t:
            return None

        # 1. Explicit web shortcut
        for key, spec in self.web.items():
            for alias in spec.get("aliases", []):
                if t == alias or _contains(t, alias) or _contains(alias, t):
                    return Target("url", key.replace("_", " "),
                                  self._browser_argv(spec["url"], browser, want_window), 1.0)

        # 2. Explicit app alias
        for key, spec in self.apps.items():
            for alias in spec.get("aliases", []):
                if t == alias:
                    return self._app_target(key, spec, want_window, 1.0)
        for key, spec in self.apps.items():
            for alias in spec.get("aliases", []):
                if _contains(t, alias) or _contains(alias, t):
                    return self._app_target(key, spec, want_window, 0.9)

        # 3. Bare URL or domain
        if t.startswith(("http://", "https://")):
            return Target("url", t, self._browser_argv(t, browser, want_window), 1.0)
        if "." in t and " " not in t and not t.endswith("."):
            url = f"https://{t}"
            return Target("url", t, self._browser_argv(url, browser, want_window), 0.8)

        # 4. Fuzzy against every installed desktop entry
        #
        # Two guards, both earned. WRatio's partial-ratio branch rewards a short
        # candidate found anywhere inside a long query, so an entire spoken
        # sentence scores highly against almost any app name — measured here,
        # "university free source articles on smoother and more human sounding
        # ai voice agents" matched "passwords and keys" at 85.5, and HIVE opened
        # Seahorse. So: only phrases short enough to plausibly *be* a name are
        # scanned at all, and the whole-string ratio has to agree with WRatio
        # before a match counts. A name misheard by a syllable still passes both;
        # a sentence passes neither.
        index = self._index_desktop()
        if index and len(t.split()) <= 5:
            for name, score, _ in process.extract(
                    t, index.keys(), scorer=fuzz.WRatio, score_cutoff=76, limit=5):
                if fuzz.ratio(t, name) < 58:
                    continue        # agrees only on a fragment, not on the name
                return Target("app", name.title(),
                              ["gio", "launch", str(index[name])], score / 100.0)

        # 5. Fuzzy against web aliases as a last resort
        #
        # Guarded identically to the desktop scan above, and for the same
        # measured reason: unguarded, "the difference between a comma and a
        # semicolon" scored past 84 against a Google Docs alias and would have
        # opened a document.
        alias_map = {a: k for k, s in self.web.items() for a in s.get("aliases", [])}
        if alias_map and len(t.split()) <= 5:
            for alias, score, _ in process.extract(
                    t, alias_map.keys(), scorer=fuzz.WRatio, score_cutoff=84, limit=5):
                if fuzz.ratio(t, alias) < 58:
                    continue
                key = alias_map[alias]
                return Target("url", key.replace("_", " "),
                              self._browser_argv(self.web[key]["url"], browser, want_window),
                              score / 100.0)
        return None

    def _app_target(self, key: str, spec: dict, want_window: bool, score: float) -> Target:
        label = spec.get("aliases", [key])[0]
        if want_window and spec.get("new_window"):
            argv = list(spec["new_window"])
            if shutil.which(argv[0]):
                return Target("app", label, argv, score)
        for argv in (spec.get("exec"), ):
            if argv and shutil.which(argv[0]):
                return Target("app", label, list(argv), score)
        desktop = spec.get("desktop")
        if desktop:
            for d in DESKTOP_DIRS:
                p = d / desktop
                if p.exists():
                    return Target("app", label, ["gio", "launch", str(p)], score)
        if spec.get("exec"):
            return Target("app", label, list(spec["exec"]), score * 0.7)
        return Target("app", label, ["gtk-launch", desktop or key], score * 0.6)

    def _browser_argv(self, url: str, browser: str | None, want_window: bool) -> list[str]:
        name = browser or self.browsers.get("default", "chrome")
        spec = self.browsers.get(name) or self.browsers.get("chrome") or {}
        base = spec.get("new_window" if want_window else "exec") or ["xdg-open"]
        argv = list(base)
        if not shutil.which(argv[0]):
            # Fall back through the other configured browser, then the portal.
            for alt in self.browsers.values():
                cand = list(alt.get("exec") or [])
                if cand and shutil.which(cand[0]):
                    argv = cand
                    break
            else:
                argv = ["xdg-open"]
        return argv + [url]

    # -- execution ----------------------------------------------------------
    def launch(self, target: Target) -> tuple[bool, str]:
        try:
            subprocess.Popen(
                target.argv,
                start_new_session=True,      # survive HIVE exiting
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                env={**os.environ},
            )
            self.bus.log(f"launch {' '.join(target.argv[:3])}", "action")
            return True, target.label
        except FileNotFoundError:
            return False, f"{target.argv[0]} is not installed"
        except Exception as exc:
            return False, str(exc)

    def open_path(self, path: Path) -> tuple[bool, str]:
        if not path.exists():
            return False, "no such file"
        return self.launch(Target("file", path.name, ["gio", "open", str(path)], 1.0))
