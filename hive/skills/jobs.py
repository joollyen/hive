"""Job search — filtered search URLs, and reading a local resume.

**What this cannot do, stated up front.** HIVE has no network access, by design.
It cannot read listings off Indeed or ZipRecruiter. Doing so would need scraping
(fragile, and against both sites' terms) or their paid APIs with credentials
sitting on this machine. Either would break the "nothing leaves this box"
guarantee that the rest of the architecture is built on.

What it does instead is the half that does not need the network, and does it
exactly: it constructs the **precise filtered search URL** for each site —
keyword, location, radius, date-posted — and opens them. "Overnight help desk,
San Diego, 25 miles, last week" becomes one click per site rather than six form
fields. It also reads a resume PDF locally and can compare it against a job
description you paste or dictate.

So the loop closes with Jay as the step that crosses the network. That is a real
limitation and it is named rather than papered over.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus


@dataclass
class SearchTarget:
    site: str
    label: str
    url: str


def _q(text: str) -> str:
    return quote_plus(text.strip())


def build_urls(role: str, location: str = "", radius: int = 25,
               days: int = 7, sites: list[str] | None = None) -> list[SearchTarget]:
    """One pre-filtered search URL per site.

    Each site spells its filters differently — Indeed uses `fromage` in days,
    LinkedIn wants `f_TPR=r<seconds>`, Glassdoor uses a different radius scale.
    Getting these right is the whole value of the skill: a wrong parameter looks
    like it worked and silently returns unfiltered results.
    """
    sites = sites or ["indeed", "ziprecruiter"]
    role = role or "help desk"
    out: list[SearchTarget] = []

    for site in sites:
        if site == "indeed":
            url = (f"https://www.indeed.com/jobs?q={_q(role)}"
                   f"&l={_q(location)}&radius={radius}&fromage={min(days, 30)}&sort=date")
            out.append(SearchTarget("indeed", "Indeed", url))
        elif site == "ziprecruiter":
            url = (f"https://www.ziprecruiter.com/jobs-search?search={_q(role)}"
                   f"&location={_q(location)}&radius={radius}&days={min(days, 30)}")
            out.append(SearchTarget("ziprecruiter", "ZipRecruiter", url))
        elif site == "linkedin":
            # LinkedIn takes a time window in seconds, not days.
            url = (f"https://www.linkedin.com/jobs/search/?keywords={_q(role)}"
                   f"&location={_q(location)}&distance={radius}&f_TPR=r{days * 86400}"
                   f"&sortBy=DD")
            out.append(SearchTarget("linkedin", "LinkedIn", url))
        elif site == "glassdoor":
            url = (f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={_q(role)}"
                   f"&locKeyword={_q(location)}&fromAge={min(days, 30)}")
            out.append(SearchTarget("glassdoor", "Glassdoor", url))
    return out


def spoken_summary(role: str, location: str, radius: int, days: int,
                   targets: list[SearchTarget]) -> str:
    where = f" within {radius} miles of {location.title()}" if location else ""
    when = "today" if days <= 1 else f"the last {days} days"
    names = " and ".join(t.label for t in targets) or "the job boards"
    return f"Opening {names} for {role}{where}, posted in {when}."


# --- resume -----------------------------------------------------------------

RESUME_HINTS = ("resume", "cv", "curriculum")


def find_resume(search_roots: list[Path] | None = None) -> Path | None:
    """Locate the most recently modified resume-looking PDF under $HOME."""
    roots = search_roots or [Path.home() / "Documents", Path.home() / "Downloads",
                             Path.home() / "Desktop", Path.home()]
    best: tuple[float, Path] | None = None
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for p in root.glob("*.pdf"):
                if any(h in p.stem.lower() for h in RESUME_HINTS):
                    mt = p.stat().st_mtime
                    if best is None or mt > best[0]:
                        best = (mt, p)
        except OSError:
            continue
    return best[1] if best else None


def read_pdf(path: Path, max_chars: int = 12000) -> str:
    """Extract text from a PDF. pypdf first, poppler as a fallback."""
    path = Path(path)
    if not path.is_file():
        return ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return re.sub(r"\n{3,}", "\n\n", text)[:max_chars]
    except Exception:
        pass
    if shutil.which("pdftotext"):
        try:
            out = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                 capture_output=True, text=True, timeout=15)
            if out.returncode == 0:
                return re.sub(r"\n{3,}", "\n\n", out.stdout)[:max_chars]
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


# Skills a help-desk / IT-support listing typically asks for. Used to give a
# concrete, checkable read of a resume rather than a vague impression.
_SIGNALS = {
    "A+": r"\bA\+|\bCompTIA\b",
    "Network+": r"\bNetwork\s*\+",
    "Security+": r"\bSecurity\s*\+",
    "eJPT": r"\beJPT\b",
    "ticketing": r"\b(ticket|ServiceNow|Jira|Zendesk|Freshdesk)\b",
    "Active Directory": r"\bActive Directory\b|\bAD\b",
    "Windows": r"\bWindows\b",
    "Linux": r"\bLinux\b",
    "networking": r"\b(TCP/IP|DNS|DHCP|VLAN|subnet)\b",
    "customer-facing": r"\b(customer|client|user support|help ?desk|valet)\b",
    "scripting": r"\b(Python|Bash|PowerShell)\b",
    "remote support": r"\b(remote|VPN|RDP|SSH)\b",
}


def profile_resume(text: str) -> dict:
    """What the resume demonstrably contains, as checkable signals."""
    found, missing = [], []
    for name, pat in _SIGNALS.items():
        (found if re.search(pat, text, re.I) else missing).append(name)
    return {"found": found, "missing": missing, "chars": len(text)}
