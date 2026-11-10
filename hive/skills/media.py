"""Media transport control over MPRIS (D-Bus).

Spotify, VLC, mpv, Chrome and Firefox all expose the standard MPRIS2 interface,
so one implementation covers everything Jay actually plays audio through. Driven
via `gdbus`, which ships with GLib and is already installed — no playerctl, no
extra Python dependency, and nothing to keep in sync with a player's own API.

What this can and cannot do is worth stating plainly, because the honest failure
message depends on it:

  CAN   play, pause, next, previous, stop, read what is currently playing
  CAN   hand a player a URI it already understands
  CANNOT search a library or resolve "my Deep Focus playlist" to something
        playable — that needs the Spotify Web API, an account token, and network
        access, none of which HIVE has by design.

So a request to play a named playlist is answered with the reason and a concrete
alternative, never with a freeze or a guess.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

BUS_PREFIX = "org.mpris.MediaPlayer2."
OBJ = "/org/mpris/MediaPlayer2"
PLAYER_IFACE = "org.mpris.MediaPlayer2.Player"
PROPS_IFACE = "org.freedesktop.DBus.Properties"

# Friendly names for the players Jay has installed.
FRIENDLY = {
    "spotify": "Spotify", "vlc": "VLC", "mpv": "mpv",
    "chromium": "Chrome", "chrome": "Chrome", "firefox": "Firefox",
}


@dataclass
class NowPlaying:
    title: str = ""
    artist: str = ""
    album: str = ""
    player: str = ""
    status: str = ""

    @property
    def spoken(self) -> str:
        if not self.title:
            return f"{FRIENDLY.get(self.player, self.player)} isn't playing anything."
        who = f" by {self.artist}" if self.artist else ""
        verb = "Playing" if self.status.lower() == "playing" else "Paused on"
        return f"{verb} {self.title}{who}."


def _gdbus(args: list[str], timeout: float = 3.0) -> str | None:
    try:
        out = subprocess.run(["gdbus"] + args, capture_output=True, text=True,
                             timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def players() -> list[str]:
    """Short names of every MPRIS player currently on the session bus."""
    out = _gdbus(["call", "--session", "--dest", "org.freedesktop.DBus",
                  "--object-path", "/org/freedesktop/DBus",
                  "--method", "org.freedesktop.DBus.ListNames"])
    if not out:
        return []
    found = re.findall(r"org\.mpris\.MediaPlayer2\.([A-Za-z0-9_.-]+)", out)
    seen, ordered = set(), []
    for name in found:
        short = name.split(".")[0].lower()
        if short not in seen:
            seen.add(short)
            ordered.append(short)
    return ordered


def _prop(player: str, name: str) -> str | None:
    return _gdbus(["call", "--session", "--dest", f"{BUS_PREFIX}{player}",
                   "--object-path", OBJ, "--method", f"{PROPS_IFACE}.Get",
                   PLAYER_IFACE, name])


def status(player: str) -> str:
    raw = _prop(player, "PlaybackStatus") or ""
    m = re.search(r"'([^']+)'", raw)
    return m.group(1) if m else ""


def active_player(preferred: str | None = None) -> str | None:
    """Pick which player a bare "pause" refers to.

    A player that is actually playing wins over one that is merely running —
    otherwise "pause" hits a backgrounded browser tab instead of the music.
    """
    found = players()
    if not found:
        return None
    if preferred:
        for p in found:
            if preferred.lower() in p:
                return p
    for p in found:
        if status(p).lower() == "playing":
            return p
    for p in found:
        if status(p).lower() == "paused":
            return p
    return found[0]


def now_playing(player: str | None = None) -> NowPlaying | None:
    p = player or active_player()
    if not p:
        return None
    raw = _prop(p, "Metadata") or ""
    def field(key: str) -> str:
        m = re.search(rf"'{key}':\s*<'([^']*)'>", raw)
        if m:
            return m.group(1)
        m = re.search(rf"'{key}':\s*<\['([^']*)'", raw)   # arrays: artist, albumArtist
        return m.group(1) if m else ""
    return NowPlaying(field("xesam:title"), field("xesam:artist"),
                      field("xesam:album"), p, status(p))


def _command(player: str, method: str) -> bool:
    return _gdbus(["call", "--session", "--dest", f"{BUS_PREFIX}{player}",
                   "--object-path", OBJ, "--method",
                   f"{PLAYER_IFACE}.{method}"]) is not None


def play(player: str | None = None) -> tuple[bool, str]:
    p = active_player(player)
    if not p:
        return False, "no player is running"
    if not _command(p, "Play"):
        return False, f"{FRIENDLY.get(p, p)} refused the command"
    np_ = now_playing(p)
    return True, (np_.spoken if np_ else f"Playing on {FRIENDLY.get(p, p)}.")


def pause(player: str | None = None) -> tuple[bool, str]:
    p = active_player(player)
    if not p:
        return False, "no player is running"
    return (True, "Paused.") if _command(p, "Pause") else (False, "the player refused")


def playpause(player: str | None = None) -> tuple[bool, str]:
    p = active_player(player)
    if not p:
        return False, "no player is running"
    was = status(p).lower()
    if not _command(p, "PlayPause"):
        return False, "the player refused"
    if was == "playing":
        return True, "Paused."
    np_ = now_playing(p)
    return True, (np_.spoken if np_ else "Playing.")


def skip(player: str | None = None, backwards: bool = False) -> tuple[bool, str]:
    p = active_player(player)
    if not p:
        return False, "no player is running"
    if not _command(p, "Previous" if backwards else "Next"):
        return False, "the player refused"
    # Metadata lags the transport change by a moment; a short settle avoids
    # announcing the track that was just skipped away from.
    import time
    time.sleep(0.35)
    np_ = now_playing(p)
    return True, (np_.spoken if np_ else "Skipped.")


def stop(player: str | None = None) -> tuple[bool, str]:
    p = active_player(player)
    if not p:
        return False, "no player is running"
    return (True, "Stopped.") if _command(p, "Stop") else (False, "the player refused")


def open_uri(uri: str, player: str | None = None) -> tuple[bool, str]:
    """Hand a player a URI it already understands (e.g. spotify:search:jazz)."""
    p = active_player(player)
    if not p:
        return False, "no player is running"
    ok = _gdbus(["call", "--session", "--dest", f"{BUS_PREFIX}{p}",
                 "--object-path", OBJ, "--method", f"{PLAYER_IFACE}.OpenUri", uri])
    return (ok is not None), uri


def search_uri(query: str) -> str:
    """A Spotify search URI for a spoken phrase.

    The honest middle ground for "play my Deep Focus playlist": HIVE cannot
    resolve the name to a playlist, but it can put the search in front of Jay so
    the next action is one click rather than a dead end.
    """
    return "spotify:search:" + re.sub(r"\s+", "%20", query.strip())
