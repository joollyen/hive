"""Read-only system queries.

Gated behind spoken confirmation by configuration, even though every call here is
non-mutating. That is the operator's choice and the gate is enforced upstream in
the objective hemisphere; this module simply refuses to contain anything that
writes, so a bug in the gate cannot become a bug in the filesystem.

Values come back as (spoken, precise) pairs. Speech wants "about four hundred
gigabytes free"; the vault log wants "412.7 GB". Formatting them together keeps
those two from drifting apart.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Reading:
    spoken: str
    precise: str


def _round_spoken(value: float, unit: str) -> str:
    """Numbers a person would actually say out loud."""
    if value >= 100:
        return f"about {round(value / 10) * 10:.0f} {unit}"
    if value >= 10:
        return f"about {value:.0f} {unit}"
    return f"{value:.1f} {unit}"


def disk(path: str = "/") -> Reading:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    total_gb = usage.total / 1e9
    pct = 100.0 * usage.used / usage.total
    return Reading(
        f"{_round_spoken(free_gb, 'gigabytes')} free, {pct:.0f} percent used",
        f"{free_gb:.1f}GB free of {total_gb:.1f}GB ({pct:.1f}% used)",
    )


def memory() -> Reading:
    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        try:
            info[key] = int(rest.strip().split()[0]) * 1024
        except (ValueError, IndexError):
            continue
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    used_pct = 100.0 * (total - avail) / total if total else 0.0
    return Reading(
        f"{_round_spoken(avail / 1e9, 'gigabytes')} available, {used_pct:.0f} percent in use",
        f"{avail/1e9:.1f}GB available of {total/1e9:.1f}GB ({used_pct:.1f}% used)",
    )


def cpu() -> Reading:
    load1, load5, _ = os.getloadavg()
    ncpu = os.cpu_count() or 1
    pct = 100.0 * load1 / ncpu
    return Reading(
        f"load is {load1:.1f}, roughly {pct:.0f} percent across {ncpu} threads",
        f"load {load1:.2f}/{load5:.2f} on {ncpu} threads ({pct:.0f}%)",
    )


def gpu() -> Reading | None:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3.0,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, util, used, total, temp = [x.strip() for x in out.stdout.strip().split(",")]
        used_gb, total_gb = float(used) / 1024, float(total) / 1024
        return Reading(
            f"the {name.replace('NVIDIA GeForce ', '')} is at {util} percent, "
            f"{used_gb:.1f} of {total_gb:.0f} gigabytes of video memory, {temp} degrees",
            f"{name} util={util}% vram={used_gb:.2f}/{total_gb:.1f}GB temp={temp}C",
        )
    except Exception:
        return None


def uptime() -> Reading:
    secs = float(Path("/proc/uptime").read_text().split()[0])
    days, rem = divmod(int(secs), 86400)
    hours, rem = divmod(rem, 3600)
    mins = rem // 60
    if days:
        spoken = f"{days} day{'s' if days != 1 else ''} and {hours} hours"
    elif hours:
        spoken = f"{hours} hour{'s' if hours != 1 else ''} and {mins} minutes"
    else:
        spoken = f"{mins} minutes"
    return Reading(f"up for {spoken}", f"{days}d {hours}h {mins}m")


def battery() -> Reading | None:
    base = Path("/sys/class/power_supply")
    if not base.is_dir():
        return None
    for bat in sorted(base.glob("BAT*")):
        try:
            pct = int((bat / "capacity").read_text().strip())
            status = (bat / "status").read_text().strip().lower()
            return Reading(f"battery is at {pct} percent and {status}",
                           f"{pct}% ({status})")
        except OSError:
            continue
    return None


def temperature() -> Reading | None:
    g = gpu()
    if g:
        return Reading(g.spoken.split(", ")[-1] + " on the G P U", g.precise)
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            c = int((zone / "temp").read_text().strip()) / 1000.0
            return Reading(f"about {c:.0f} degrees", f"{c:.1f}C")
        except OSError:
            continue
    return None


def snapshot() -> dict[str, Reading]:
    """Everything at once, for the objective hemisphere's grounded-fact set."""
    out: dict[str, Reading] = {
        "disk": disk(), "memory": memory(), "cpu": cpu(), "uptime": uptime(),
    }
    for name, fn in (("gpu", gpu), ("battery", battery)):
        val = fn()
        if val is not None:
            out[name] = val
    return out


def answer(question: str) -> Reading | None:
    """Route a spoken system question to the right reading."""
    q = question.lower()
    if any(w in q for w in ("disk", "storage", "space", "drive")):
        return disk()
    if any(w in q for w in ("gpu", "graphics", "vram", "video memory", "card")):
        return gpu()
    if any(w in q for w in ("ram", "memory")):
        return memory()
    if any(w in q for w in ("cpu", "processor", "load")):
        return cpu()
    if "batter" in q:
        return battery()
    if any(w in q for w in ("temp", "hot", "warm", "thermal")):
        return temperature()
    if "uptime" in q or "how long" in q:
        return uptime()
    return None
