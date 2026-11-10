"""The query itself: text in an editor, parsed into tasks.

HIVE proper is built for the moment — one utterance, one turn, one answer, and
nothing carried between them except memory. That is the right shape for "HIVE,
open Chrome" and the wrong shape for "here are three things, do them while I work
on something else". A queue is a different object: it exists before the work
starts, it outlives any single turn, and its whole value is that its state is
written down rather than held in a model's context.

So the source of truth is a plain text buffer, one task per line, and everything
downstream is derived from it. That choice is deliberate and it is the same one
the vault makes: a format a person can read, edit, retype and delete, with no
database and no hidden state. A task list you cannot fix in a text editor is one
you stop trusting the moment it is wrong.

Parsing is forgiving about how a list gets written, because the point of typing
into a pane rather than speaking is that you write it the way you think:

    1. hive I need you to find me research articles on neural networks
    2) hive research keyboards similar to the corsair vanguard
    - hive open youtube and search betterminecraft+ hardcore gameplay
    # this line is a comment and is not a task
    hive check the disk, \\
      and tell me what's using it          <- continued onto the next line

Enumerators are stripped rather than honoured. The order is the order of the
lines; a list numbered "1, 2, 2, 5" after some editing still runs top to bottom,
which is what someone who just deleted a line expects.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# Status vocabulary. `incomplete` is the one Jay named — the tab things go to when
# HIVE could not finish them — and it is deliberately distinct from `failed`,
# because "the watchdog gave up on it" and "it ran and returned an error" want
# different sentences in the note that gets written afterwards.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
INCOMPLETE = "incomplete"      # stalled out, or refused, or ended by hand
SKIPPED = "skipped"            # never attempted — the query ended before it ran

UNFINISHED = (INCOMPLETE, SKIPPED)

# Enumerators and bullets people actually type at the head of a list item.
_ENUMERATOR = re.compile(r"^\s*(?:\(?\d{1,3}[.)\]]|\d{1,3}\s*[-–]|[-*•·▸>])\s+")
_COMMENT = re.compile(r"^\s*(?:#|//)")
_CONTINUES = re.compile(r"\\\s*$")

MAX_TASKS = 60
MAX_TASK_CHARS = 600


@dataclass
class QueryTask:
    """One item of work, and everything that happened to it.

    State lives here rather than in a model's context for the reason planner.py
    already documents: τ-Voice measured agents completing part of a multi-step
    request and reporting success anyway. A queue makes that failure mode worse,
    not better — it is longer, and nobody is watching it. So each task carries
    its own status, its own spoken record and its own reason for failing, and the
    note written to the vault afterwards is composed from these fields rather
    than from anything HIVE remembers about them.
    """

    text: str
    index: int                       # 1-based, as displayed
    line: int = 0                    # source line in the editor buffer
    status: str = PENDING
    reason: str = ""                 # why it is not done, in plain speech
    spoken: str = ""                 # what HIVE actually said while doing it
    actions: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0
    note: str = ""                   # filename of the markdown log, once written

    @property
    def duration(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    @property
    def ok(self) -> bool:
        return self.status == DONE

    @property
    def title(self) -> str:
        """A short human label — the editor gutter and the .md filename use it.

        The leading "hive" is stripped because every line in a query tends to
        start with it (people write the list the way they would say it) and a
        folder of notes all called "Hive Find Me…" is a folder you cannot scan.
        """
        t = re.sub(r"^\s*(?:hey\s+)?hive\b[,:]?\s*", "", self.text, flags=re.I)
        t = re.sub(r"^\s*(?:i\s+need\s+you\s+to|i\s+want\s+you\s+to|can\s+you|"
                   r"could\s+you|please)\s+", "", t, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip(" .,:;-")
        return t or self.text.strip()

    def short(self, width: int = 46) -> str:
        t = self.title
        return t if len(t) <= width else t[:width].rsplit(" ", 1)[0] + "…"


def parse(buffer: str) -> list[QueryTask]:
    """Turn an editor buffer into an ordered task list."""
    tasks: list[QueryTask] = []
    pending: list[str] = []
    start_line = 0

    def flush() -> None:
        nonlocal pending, start_line
        if not pending:
            return
        text = " ".join(p.strip() for p in pending).strip()
        text = _ENUMERATOR.sub("", text).strip()
        pending = []
        if len(text) < 3:
            return
        tasks.append(QueryTask(text=text[:MAX_TASK_CHARS],
                               index=len(tasks) + 1, line=start_line))

    for n, raw in enumerate((buffer or "").splitlines(), start=1):
        if _COMMENT.match(raw):
            continue
        if not raw.strip():
            flush()
            continue
        if not pending:
            start_line = n
        if _CONTINUES.search(raw):
            pending.append(_CONTINUES.sub("", raw))
            continue
        pending.append(raw)
        flush()
        if len(tasks) >= MAX_TASKS:
            break
    flush()
    return tasks[:MAX_TASKS]


def summarise(tasks: list[QueryTask]) -> str:
    """One line of plain English about how a finished query went.

    Enumerates the unfinished ones by name. A queue that half-completed must say
    which half, in the same breath as the count — the closing report is the only
    moment anyone is listening, and "three of five" without the names of the
    other two is a number nobody can act on.
    """
    done = [t for t in tasks if t.status == DONE]
    bad = [t for t in tasks if t.status in UNFINISHED]
    if not tasks:
        return "There was nothing in the query."
    if not bad:
        return (f"Query's done. All {len(done)} of them."
                if len(done) != 1 else "Query's done. That was the only one.")
    names = ", ".join(t.short(30) for t in bad)
    if not done:
        return (f"None of the {len(tasks)} went through. "
                f"They're all in the incomplete tab: {names}.")
    return (f"Query's done. {len(done)} of {len(tasks)} finished. "
            f"{len(bad)} went to the incomplete tab: {names}.")
