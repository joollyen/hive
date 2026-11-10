"""Long-term memory — the Obsidian vault under HIVE/.

Design constraints that come from it being a *vault* and not a database:

  - Everything is markdown a human can read, edit and delete. If HIVE writes
    something wrong, Jay fixes it in Obsidian, not by clearing a cache.
  - Links are [[wikilinks]], so Obsidian's graph view actually shows the shape of
    what has been discussed.
  - Files are never rewritten wholesale. Topic notes accumulate dated entries by
    appending under a marker, so hand-edits above it survive consolidation.
  - Numeric folder prefixes force a stable sort order in the file explorer.

Consolidation runs once, at session close, on the heavy model. That is the one
place in the system where latency genuinely does not matter — the user has
already said goodnight — so it gets the good model and a long generation budget.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SESSIONS = "00_Sessions"
TOPICS = "10_Topics"
INBOX = "20_Inbox"
OPERATOR = "30_Operator"
WRITEUPS = "50_Writeups"
# HIVE-Q's record. Named plainly rather than with a numeric prefix like its
# neighbours because it is the one folder Jay asked for by name, and because it
# is written to continuously during a run rather than once at the end — it reads
# more like a log directory than a section of the notebook.
QUERY_TASKS = "query_tasks"
STATE = "_state"

_SAFE = re.compile(r"[^\w\s&()-]+")
_APPEND_MARKER = "<!-- hive:append -->"

# Models routinely return topic labels with the spaces stripped —
# "diskspacecheck", "postejptfocus". Left alone those become their own notes and
# sit beside the properly-spaced version forever.
_RUNON = re.compile(
    r"(?i)(disk|space|check|exam|prep|preparation|study|focus|review|notes?|"
    r"practice|practise|knowledge|application|scenario|scenarios|revision|"
    r"session|report|resume|project|lab|setup|config|system|memory|voice|post|"
    r"real|world|target|targeted|consolidation|readiness|problem|solving|key|"
    r"concepts?|ejpt|ecppt|hive)")


def _split_runon(text: str) -> str:
    """Recover word boundaries in a squashed label, if we can do it cleanly."""
    if " " in text.strip() or len(text) < 9:
        return text
    parts, pos = [], 0
    for m in _RUNON.finditer(text):
        if m.start() != pos:          # unrecognised span; refuse rather than mangle
            return text
        parts.append(m.group(0))
        pos = m.end()
    return " ".join(parts) if pos == len(text) and len(parts) > 1 else text


def _sentence(text: str) -> str:
    """First letter up, everything else untouched, one full stop at the end."""
    t = (text or "").strip()
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?" else t + "."


def slugify_title(text: str) -> str:
    """A filename Obsidian is happy with and a person can still read."""
    t = _SAFE.sub("", text).strip()
    t = re.sub(r"\s+", " ", t)
    t = _split_runon(t)
    return (t[:60] or "Untitled").title()


@dataclass
class Note:
    path: Path
    title: str
    created: bool


class LongTermMemory:
    def __init__(self, cfg, bus):
        self.cfg, self.bus = cfg, bus
        self.root: Path = cfg.hive_dir
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # The session's note, allocated on the first write and reused by every
        # write after it. This is what makes the transcript safe: it can be put
        # on disk the moment there is one, and the summary written over the top
        # of it later, without ending up with two notes for one conversation.
        self._session_note: Path | None = None

    # -- structure ----------------------------------------------------------
    def ensure(self) -> None:
        """Create the folder skeleton and seed operator knowledge on first run."""
        for sub in (SESSIONS, TOPICS, INBOX, OPERATOR, WRITEUPS, QUERY_TASKS, STATE):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        index = self.root / "HIVE.md"
        if not index.exists():
            index.write_text(_INDEX_TEMPLATE, encoding="utf-8")
        self._seed_operator()

    def _seed_operator(self) -> None:
        """Preloaded working knowledge — HIVE's equivalent of JARVIS's briefing.

        Seeded from what the vault already establishes rather than invented, and
        written once. If Jay edits these, they are never overwritten.
        """
        for name, body in _OPERATOR_SEED.items():
            p = self.root / OPERATOR / f"{name}.md"
            if not p.exists():
                p.write_text(body, encoding="utf-8")

    # -- state --------------------------------------------------------------
    def _state_path(self, name: str) -> Path:
        return self.root / STATE / f"{name}.json"

    def load_state(self, name: str, default: dict | None = None) -> dict:
        p = self._state_path(name)
        if not p.exists():
            return dict(default or {})
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return dict(default or {})

    def save_state(self, name: str, data: dict) -> None:
        try:
            self._state_path(name).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self.bus.log(f"state write failed: {exc}", "warn")

    # -- quick capture ------------------------------------------------------
    def add_inbox(self, content: str) -> Note:
        """Append a timestamped bullet to today's inbox note.

        This is the "HIVE, note that..." path. It has to be fast and lossless —
        no summarising, no interpretation, exactly what was said.
        """
        day = datetime.now().strftime("%Y-%m-%d")
        path = self.root / INBOX / f"{day}.md"
        created = not path.exists()
        if created:
            path.write_text(
                f"---\ntags: [hive/inbox]\ncreated: {day}\n---\n\n"
                f"# Inbox — {day}\n\nCaptured by [[HIVE]] during voice sessions.\n\n",
                encoding="utf-8",
            )
        stamp = datetime.now().strftime("%H:%M")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"- **{stamp}** — {content.strip()}\n")
        return Note(path, f"Inbox {day}", created)

    # -- topics -------------------------------------------------------------
    def existing_topics(self) -> list[str]:
        d = self.root / TOPICS
        return [p.stem for p in d.glob("*.md")] if d.is_dir() else []

    def resolve_topic(self, topic: str, threshold: int = 84) -> str:
        """Reuse an existing topic note when one is close enough.

        Without this the vault accumulates "Disk Space Check", "Diskspace Check"
        and "Diskspacecheck" as three separate notes, because the summariser
        phrases the same subject slightly differently every session. Topic notes
        are supposed to accumulate history; splitting them defeats the point.
        """
        title = slugify_title(topic)
        existing = self.existing_topics()
        if not existing or title in existing:
            return title
        from rapidfuzz import fuzz, process
        # Compare on letters only, so spacing and case cannot create a new note.
        def key(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())
        lookup = {key(e): e for e in existing}
        match = process.extractOne(key(title), lookup.keys(),
                                   scorer=fuzz.ratio, score_cutoff=threshold)
        return lookup[match[0]] if match else title

    def append_topic(self, topic: str, entry: str, session_link: str = "") -> Note:
        title = self.resolve_topic(topic)
        path = self.root / TOPICS / f"{title}.md"
        created = not path.exists()
        if created:
            path.write_text(
                f"---\ntags: [hive/topic]\ncreated: {datetime.now():%Y-%m-%d}\n---\n\n"
                f"# {title}\n\n"
                f"Topic note maintained by [[HIVE]].\n\n## Log\n\n{_APPEND_MARKER}\n",
                encoding="utf-8",
            )
        text = path.read_text(encoding="utf-8")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        link = f" — {session_link}" if session_link else ""
        block = f"- **{stamp}**{link} — {entry.strip()}\n"
        # Insert above the marker so anything a human added below is preserved.
        if _APPEND_MARKER in text:
            text = text.replace(_APPEND_MARKER, block + _APPEND_MARKER)
        else:
            text += "\n" + block
        path.write_text(text, encoding="utf-8")
        return Note(path, title, created)

    # -- search -------------------------------------------------------------
    def search(self, query: str, limit: int = 6) -> list[tuple[Path, str]]:
        """Keyword search across the HIVE folder.

        Deliberately not embeddings. At this vault's size a scored substring match
        returns in single-digit milliseconds and never surprises the user with a
        semantically-adjacent but wrong note — and a wrong recall spoken aloud
        with confidence is worse than no recall at all.
        """
        terms = [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]
        if not terms:
            return []
        hits: list[tuple[float, Path, str]] = []
        for path in self.root.rglob("*.md"):
            if STATE in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            low = text.lower()
            score = sum(low.count(t) for t in terms)
            if not score:
                continue
            # Filename matches are a much stronger signal than body mentions.
            score += 5 * sum(1 for t in terms if t in path.stem.lower())
            hits.append((score, path, self._excerpt(text, terms)))
        hits.sort(key=lambda h: -h[0])
        return [(p, ex) for _s, p, ex in hits[:limit]]

    @staticmethod
    def _excerpt(text: str, terms: list[str], width: int = 220) -> str:
        low = text.lower()
        for t in terms:
            i = low.find(t)
            if i >= 0:
                start = max(0, i - width // 3)
                return re.sub(r"\s+", " ", text[start:start + width]).strip()
        return re.sub(r"\s+", " ", text[:width]).strip()

    # -- session consolidation ---------------------------------------------
    def write_session(self, stm, actions: list[str], summary: str,
                      topics: list[str],
                      calibrations: list[tuple[str, float, int]] | None = None) -> Note:
        """Write this session's note. Safe to call more than once.

        Called twice on a normal shutdown — see Runtime.consolidate. The first
        call lands the transcript in the vault immediately, before the language
        model is asked for anything; the second replaces it with the same note
        plus a summary. Two calls, one file, because the second one reuses the
        path the first allocated.

        That ordering is the whole point. Generating the summary takes between
        nine and fifty-seven seconds on this box — measured across the sessions
        in .logs/hive.log — and for a long time the transcript existed only in
        the memory of a process that was trying to exit. Seven sessions in that
        log start consolidating and never finish. The conversation was gone, and
        it had been sitting in RAM, fully formed, the entire time.
        """
        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        first_write = self._session_note is None

        if first_write:
            folder = self.root / SESSIONS / day
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"{now:%H%M} Session"
            path = folder / f"{stem}.md"
            n = 2
            while path.exists():
                path = folder / f"{stem} ({n}).md"
                n += 1
            self._session_note = path
        path = self._session_note
        stem = path.stem

        mins = stm.duration / 60.0
        # Resolve through the same matcher the topic notes use, so a session's
        # wikilinks point at the notes that actually get written.
        topic_links = " ".join(f"[[{self.resolve_topic(t)}]]" for t in topics) or "—"
        lines = [
            "---",
            "tags: [hive/session]",
            f"date: {day}",
            f"time: {now:%H:%M}",
            f"duration_min: {mins:.1f}",
            f"turns: {stm.user_turns}",
            "---",
            "",
            f"# Session — {day} {now:%H:%M}",
            "",
            f"**Topics:** {topic_links}",
            f"**Duration:** {mins:.0f} min · **Exchanges:** {stm.user_turns}",
            "",
            "## Summary",
            "",
            summary.strip() or "_No summary generated._",
            "",
        ]
        if actions:
            lines += ["## Actions taken", ""] + [f"- {a}" for a in actions] + [""]

        # Which lens answered, and for how long. One session, three calibrations
        # — the whole session is recorded here regardless of which one was
        # active, and this table is the only place the split is visible.
        if calibrations:
            spent = [row for row in calibrations if row[1] >= 0.05 or row[2]]
            if spent:
                lines += ["## Calibration", "",
                          "| Lens | Time | Turns |", "| --- | --- | --- |"]
                lines += [f"| {label} | {mins:.0f} min | {turns} |"
                          for label, mins, turns in spent]
                lines.append("")

        salient = stm.salient(10)
        if salient:
            lines += ["## Salient terms", "",
                      ", ".join(f"`{t}`" for t, _ in salient), ""]

        lines += ["## Transcript", "", "```text"]
        for t in stm.transcript():
            # HIVE's own turns are tagged with the lens that produced them, so a
            # future session can tell a HackBot answer from a VOSS one.
            who = "JAY " if t.role == "user" else (
                getattr(t, "calibration", "hive") or "hive").upper().ljust(7)
            lines.append(f"{datetime.fromtimestamp(t.ts):%H:%M:%S}  {who}  {t.text}")
        lines += ["```", "", f"Linked from [[HIVE]].", ""]

        # Written to a neighbouring temp file and renamed over the target, which
        # on the same filesystem is atomic. A process killed mid-write therefore
        # leaves either the previous complete note or the new complete note, and
        # never the half-truncated transcript that would be the worst of the
        # three — a file that exists, opens, and quietly stops mid-conversation.
        body = "\n".join(lines)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
        return Note(path, stem, first_write)

    @property
    def session_note(self) -> Path | None:
        """Where this session's note is, once there is one."""
        return self._session_note

    # -- requested write-ups ------------------------------------------------
    def write_document(self, title: str, sections: list[tuple[str, str]],
                       turns: list | None = None,
                       salient: list[tuple[str, float]] | None = None,
                       actions: list[str] | None = None,
                       links: list[str] | None = None,
                       subject: str = "") -> Note:
        """Write a full document, on request, into `50_Writeups/`.

        Distinct from `write_session`, which is the automatic record kept on the
        way out. This is the one Jay asks for out loud — "make a markdown file
        with everything we talked about, in detail" — and the standing complaint
        about it is the reason for the shape below: the topic notes this vault
        accumulates are thirteen lines each, because the only prose they ever
        received was one sentence lifted from a session summary.

        So the guarantee here is structural rather than a matter of prompting.
        The transcript, the terms, the actions and the vault's own recall are
        appended verbatim from data that already exists, which means the file is
        substantial even when the language model returns two sentences or
        nothing at all. The model's prose is added *on top of* that floor, never
        in place of it. A note that says less than the conversation did is the
        failure being designed against.
        """
        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        folder = self.root / WRITEUPS
        folder.mkdir(parents=True, exist_ok=True)

        stem = slugify_title(title) or "Write-up"
        path = folder / f"{stem}.md"
        n = 2
        while path.exists():
            path = folder / f"{stem} ({n}).md"
            n += 1

        lines = [
            "---",
            "tags: [hive/writeup]",
            f"created: {day}",
            f"time: {now:%H:%M}",
            f"source: voice session {self.session_id}",
        ]
        if subject:
            lines.append(f"subject: {subject}")
        lines += [
            "---",
            "",
            f"# {stem}",
            "",
            f"*Written by [[HIVE]] on request, {now:%A %d %B %Y at %H:%M}. Everything "
            f"below comes out of that conversation — the prose is a summary of it, "
            f"and the transcript at the end is the whole of it, unedited.*",
            "",
        ]

        for heading, body in sections:
            body = (body or "").strip()
            if not body:
                continue
            lines += [f"## {heading}", "", body, ""]

        if salient:
            lines += [
                "## Terms and names that came up", "",
                "Ordered by how much of the conversation they carried.", "",
                " · ".join(f"`{t}`" for t, _ in salient), "",
            ]

        if actions:
            lines += ["## Actions taken while this was discussed", ""]
            lines += [f"- {a}" for a in actions]
            lines.append("")

        if turns:
            spoken = [t for t in turns if t.text]
            lines += [
                "## Everything said, in order", "",
                f"{len(spoken)} turns. Kept in full rather than summarised: a "
                f"write-up that drops the wording drops the part that is hard to "
                f"reconstruct later.", "",
            ]
            for t in spoken:
                who = ("**Jay**" if t.role == "user"
                       else f"**{(getattr(t, 'calibration', 'hive') or 'hive').upper()}**")
                stamp = datetime.fromtimestamp(t.ts).strftime("%H:%M:%S")
                lines.append(f"- `{stamp}` {who} — {t.text.strip()}")
            lines.append("")

        wiki = " ".join(f"[[{lnk}]]" for lnk in (links or []) if lnk)
        lines += ["---", "", f"Related: {wiki or '[[HIVE]]'}", ""]

        path.write_text("\n".join(lines), encoding="utf-8")
        return Note(path, stem, True)

    # -- HIVE-Q task log ----------------------------------------------------
    def write_query_task(self, task, session_id: str = "",
                         total: int = 0) -> Note:
        """One note per query task, written the moment the task settles.

        Written for *both* outcomes, and that is the whole design. A queue runs
        unattended, so the successes have already spoken for themselves and been
        forgotten; what Jay actually needs back at the end is the list of things
        that did not work and what state they were left in. A log that only
        records successes is a log that is silent about exactly the thing it was
        kept for.

        So a failed task's note is not a stub. It carries the reason in plain
        speech, everything HIVE said before it stopped, and any action that did
        complete — because "sent to incomplete" almost never means "nothing
        happened", it means something happened and then stopped, and the half
        that happened is what makes the task cheap to finish by hand.

        Filed one-per-file rather than appended to a daily note so that Obsidian's
        search, graph and backlinks all treat a task as a thing rather than as a
        line inside something else.
        """
        from ..query.tasklist import DONE

        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        folder = self.root / QUERY_TASKS
        folder.mkdir(parents=True, exist_ok=True)

        title = slugify_title(task.short(52)) or f"Task {task.index}"
        stem = f"{day} {now:%H%M} Q{task.index} {title}"
        path = folder / f"{stem}.md"
        n = 2
        while path.exists():
            path = folder / f"{stem} ({n}).md"
            n += 1

        complete = task.status == DONE
        spoken = (task.spoken or "").strip()
        lines = [
            "---",
            "tags: [hive/query-task]",
            f"status: {'complete' if complete else 'incomplete'}",
            f"created: {day}",
            f"time: {now:%H:%M}",
            f"task: {task.index}",
        ]
        if total:
            lines.append(f"of: {total}")
        if session_id:
            lines.append(f"query_session: {session_id}")
        lines += [
            f"duration_s: {task.duration:.1f}",
            "---",
            "",
            f"# Q{task.index} — {title}",
            "",
            f"**Status:** " + ("Complete." if complete else
                               "Incomplete — sent to the incomplete tab."),
            "",
            "**Asked:**",
            "",
            "```text",
            task.text.strip(),
            "```",
            "",
        ]

        if not complete:
            lines += [
                "## Why it didn't finish",
                "",
                # Upper-case the first letter only. `.capitalize()` lower-cases
                # everything after it, which turns "RuntimeError" into
                # "runtimeerror" — the one word in the sentence worth reading.
                _sentence(task.reason or "No reason was recorded."),
                "",
                "Fix whatever is in the way and the line above can be pasted "
                "straight back into HIVE-Q.",
                "",
            ]

        heading = ("## What HIVE said" if complete
                   else "## What HIVE got through before it stopped")
        lines += [heading, ""]
        lines += [spoken if spoken else
                  "_Nothing was spoken — the task stopped before it produced "
                  "anything._", ""]

        if task.results:
            lines += ["## What it found", ""]
            for r in task.results:
                name = (r.get("title") or r.get("url") or "").strip()
                url = (r.get("url") or "").strip()
                snip = re.sub(r"\s+", " ", (r.get("snippet") or "")).strip()
                bullet = f"- [{name}]({url})" if url and name else f"- {name or url}"
                if r.get("meta", {}).get("published"):
                    bullet += f" — {r['meta']['published']}"
                lines.append(bullet)
                if snip:
                    lines.append(f"  > {snip[:300]}")
            lines.append("")

        if task.actions:
            lines += ["## Actions taken", ""]
            lines += [f"- {a}" for a in task.actions]
            lines.append("")

        lines += [
            "---",
            "",
            f"Run by [[HIVE]] from HIVE-Q"
            + (f", query session `{session_id}`" if session_id else "") + ".",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return Note(path, stem, True)

    def link_for(self, note: Note) -> str:
        return f"[[{note.path.stem}]]"


_INDEX_TEMPLATE = """---
tags: [hive]
---

# HIVE

Long-term memory for **HIVE** — Hyper-Intelligent Voice Engine.

Everything here is written by HIVE at the end of a voice session. It is ordinary
markdown: edit it, reorganise it, delete it. HIVE appends, it does not overwrite.

## Layout

| Folder | Contents |
| --- | --- |
| `00_Sessions/` | One note per voice session, with summary and full transcript |
| `10_Topics/` | Accumulating notes per subject, appended each session |
| `20_Inbox/` | Quick captures — "HIVE, note that..." |
| `30_Operator/` | Preloaded working knowledge about Jay and his workload |
| `50_Writeups/` | Documents written on request — "write all that up" |
| `query_tasks/` | One note per HIVE-Q task, written as each one settles |
| `_state/` | Machine state (preferences, counters). Not meant to be read. |

## Recent

```dataview
TABLE duration_min AS "Min", turns AS "Turns"
FROM #hive/session
SORT date DESC, time DESC
LIMIT 12
```

_(The block above renders if the Dataview plugin is installed; harmless if not.)_
"""


# Seed notes written into the vault's operator folder on first run, then owned
# by the operator. The author's own profile is personal and is not included in
# this repository; these are templates to fill in.
_OPERATOR_SEED = {
    "Operator Profile": """---
tags: [hive/operator]
---

# Operator Profile

**Name:**
**Focus areas:**

## How to address the operator
By name. HIVE works for the operator and speaks like a competent colleague.

## Working style
- Notes in this vault are the source of truth.
""",
    "Active Fronts": """---
tags: [hive/operator]
---

# Active Fronts

The standing workstreams. Update as things land or die.
""",
    "Toolchain": """---
tags: [hive/operator]
---

# Toolchain

What is installed and used on this machine. HIVE resolves spoken app names
against this plus a live scan of installed desktop entries.

## Note on Wayland
HIVE can launch applications and open URLs and files. It cannot focus, move or
resize existing windows — Wayland does not permit that from an ordinary client.
""",
    "Workflow": """---
tags: [hive/operator]
---

# Workflow

What HIVE should make faster for you.
""",
}
