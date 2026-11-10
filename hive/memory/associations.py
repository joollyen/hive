"""Associative recall — the bridge between the vault and the live turn.

Long-term memory could already be *searched*: ask about the resume and HIVE
looks it up. What it could not do is notice. Nothing connected "check for
junior analyst listings this week" to the fact that an Acme and a Globex
were applied to three sessions ago, because nothing ever looked unless it was
asked to.

This module looks every turn. It keeps an index of the entities the vault
mentions — proper nouns, technical vocabulary, place and company names — each
with the dates it was seen on and the note it came from, and it tracks a second
thing on top: **open loops**. A line that says something was applied to, sent,
submitted or signed up for is a commitment with no recorded outcome, and a
commitment with no outcome is the single most useful thing an assistant can
raise unprompted.

Three deliberate constraints:

  - Keyword and pattern matching, not embeddings. Same reasoning as
    LongTermMemory.search: a semantically-adjacent but wrong recall, spoken
    aloud with confidence, is worse than no recall at all. Here it is worse
    still, because nobody asked for it.
  - Nothing is surfaced on a single sighting. An entity has to have been seen
    more than once, or carry an open loop, before it is worth interrupting for.
  - The index is a cache and never a source of truth. It is rebuilt from the
    markdown, per file, whenever a file's mtime changes; deleting a note in
    Obsidian deletes the memory of it.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

STATE_KEY = "associations"

# --- entity extraction ------------------------------------------------------
# Proper nouns of one to three capitalised words. Deliberately greedy about
# multiword names ("Cal Poly Pomona", "Active Directory", "Hilton Garden Inn")
# because those are exactly the ones worth remembering, and deliberately
# suspicious of single words, which are usually just the start of a sentence.
_PROPER = re.compile(r"\b([A-Z][a-zA-Z0-9&'’.-]{1,}(?:\s+[A-Z][a-zA-Z0-9&'’.-]{1,}){0,2})\b")

# Words that begin sentences and are not names, however capitalised they are.
_NOT_A_NAME = {
    "a", "an", "the", "this", "that", "these", "those", "it", "he", "she", "they",
    "we", "you", "i", "his", "her", "their", "our", "your", "my", "there", "here",
    "and", "but", "so", "if", "when", "while", "after", "before", "then", "than",
    "what", "which", "who", "whom", "whose", "where", "why", "how", "is", "was",
    "are", "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "can", "could", "will", "would", "should", "may", "might", "must",
    "no", "not", "yes", "ok", "okay", "also", "just", "still", "only", "both",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "today", "tomorrow",
    "yesterday", "tonight", "session", "summary", "topics", "actions", "inbox",
    "transcript", "duration", "exchanges", "jay", "hive", "part", "note", "notes",
    "log", "operator", "profile", "left", "right", "next", "last", "first",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # Vault scaffolding. These are labels in HIVE's own note templates, and
    # without them "Name", "Topics" and "Layout" become the vault's most
    # frequently mentioned entities.
    "name", "tags", "created", "date", "time", "turns", "contents", "layout",
    "recent", "folder", "expected", "institution", "target", "tracks", "salient",
    "terms", "linked", "graduation", "style", "working", "known", "later",
    "fall", "spring", "summer", "winter", "week", "month", "year",
    "help", "desk", "type", "path", "files", "line", "lines", "word", "words",
}

# Commitments. A line matching one of these, with an entity on it, is something
# that was started and has no recorded ending.
_LOOP_VERBS = [
    (re.compile(r"\bapplied\s+(?:to|for|at)\b", re.I), "applied to"),
    (re.compile(r"\bapplication\s+(?:to|for|at)\b", re.I), "applied to"),
    (re.compile(r"\b(?:submitted|sent)\s+(?:an?\s+)?(?:application|resume|cv)\b", re.I),
     "sent an application to"),
    (re.compile(r"\b(?:emailed|messaged|contacted|reached out to)\b", re.I), "reached out to"),
    (re.compile(r"\b(?:signed up|registered|enrolled)\s+(?:for|with|in)\b", re.I),
     "signed up for"),
    (re.compile(r"\b(?:waiting|waiting on|waiting for)\b", re.I), "were waiting on"),
    (re.compile(r"\bfollow(?:ed)?\s+up\s+(?:with|on)\b", re.I), "followed up with"),
    (re.compile(r"\b(?:booked|scheduled|registered for)\b", re.I), "booked"),
]

# ...and the things that close one.
_RESOLVED = re.compile(
    r"\b(heard back|got (?:the|an) (?:job|offer|call)|offer|rejected|turned down|"
    r"declined|interview(?:ed|ing)?|ghosted|no longer|withdrew|accepted|hired|"
    r"passed|failed|finished|completed|done with)\b", re.I)

# Dated bullet written by LongTermMemory.append_topic, and ISO dates generally.
_LINE_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# td_lists convention: 230p080226.md == 2:30pm on 08/02/26.
_TD_NAME = re.compile(r"^\d{3,4}[ap](\d{2})(\d{2})(\d{2})$")


def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def days_since(iso: str) -> int:
    try:
        then = datetime.strptime(iso, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 9999
    return max(0, (date.today() - then).days)


def human_when(days: int) -> str:
    """A spoken-form gap. Never a bare number of days past a fortnight."""
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 14:
        return "last week"
    if days < 31:
        weeks = round(days / 7)
        return f"about {weeks} weeks ago"
    if days < 75:
        return "about a month ago"
    if days < 350:
        months = round(days / 30)
        return f"about {months} months ago"
    return "a long while back"


# Terms whose casing is not recoverable from a transcript. Whisper writes them
# flat and the index only learns the right form once it has read a note that
# contains one, which the first turn of a session can beat.
_CANONICAL = {
    "ejpt": "eJPT", "ecppt": "eCPPT", "oscp": "OSCP", "crto": "CRTO",
    "comptia": "CompTIA", "tryhackme": "TryHackMe", "hackthebox": "HackTheBox",
    "nmap": "nmap", "bloodhound": "BloodHound", "impacket": "Impacket",
    "zerologon": "Zerologon", "kerberoasting": "Kerberoasting",
    "ziprecruiter": "ZipRecruiter", "linkedin": "LinkedIn",
    "hive": "HIVE", "voss": "VOSS", "hackbot": "HackBot",
}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def name_like(display: str) -> bool:
    """Is this the name of a thing, or just a word the vocabulary knows?

    The distinction decides what may be said out loud. "junior" and "analyst"
    are useful for finding the right memory and useless for reporting one: the
    line "you already applied to junior" is nonsense, while "you already
    applied to the Acme Regional" is the entire point. A name carries a capital
    somewhere or is more than one word; everything else is a topic, and topics
    are for looking things up with, not for announcing.
    """
    return any(c.isupper() for c in display) or " " in display.strip()


@dataclass
class Mention:
    date: str
    source: str
    excerpt: str
    verb: str = ""          # non-empty when this mention opened a loop
    closed: bool = False    # the line also carried a resolution


@dataclass
class Connection:
    """One thing in the current turn that the vault has seen before."""

    display: str
    days_ago: int
    times: int
    source: str
    loop_verb: str = ""
    loop_days: int = 0
    excerpt: str = ""
    # Set when this was not named in the turn at all, but was found alongside
    # something that was: "junior analyst" is what he said, the Acme is what
    # he actually needs reminding about.
    via: str = ""

    @property
    def open_loop(self) -> bool:
        return bool(self.loop_verb)

    @property
    def strength(self) -> float:
        """How much this is worth saying out loud, unprompted.

        An unclosed commitment outranks everything: it is the only kind of recall
        that can change what the operator does next. After that, repetition and
        age — something mentioned four times over a month is a thread, something
        mentioned twice yesterday is just this week's work.
        """
        s = 1.0 * min(self.times, 6)
        if self.open_loop:
            s += 8.0
            # A loop that has been open a while is the interesting one. Under
            # three days it is not stale yet, it is just recent.
            s += min(self.loop_days, 45) / 9.0
        s += min(self.days_ago, 60) / 30.0
        return s


class Associations:
    def __init__(self, cfg, bus, ltm):
        self.cfg, self.bus, self.ltm = cfg, bus, ltm
        self.enabled = bool(cfg.get("memory.associations.enabled", True))
        self._entities: dict[str, list[Mention]] = {}
        self._display: dict[str, str] = {}
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._vocab = self._known_vocabulary()

    # -- vocabulary ---------------------------------------------------------
    def _known_vocabulary(self) -> set[str]:
        """Lowercase terms worth indexing even when they are not capitalised.

        Partly seeded from the ASR hotword list, which is already this operator's
        working vocabulary — but only the tokens that could not be ordinary
        English. Splitting that list naively yields "help", "desk", "Cal", "Poly"
        and "Suite", and those went straight to the top of the index. A token
        qualifies if it carries an internal capital or a digit (eJPT, ZipRecruiter,
        CompTIA); multiword names like "Cal Poly Pomona" and "Burp Suite" are
        caught by the proper-noun pass anyway, and caught better.
        """
        raw = str(self.cfg.get("asr.hotwords", ""))
        words = {
            w.lower() for w in re.split(r"\s+", raw)
            if len(w) > 3 and (any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w))
        }
        words |= {
            "ejpt", "ecppt", "comptia", "nmap", "privesc", "kerberoasting",
            "zerologon", "impacket", "bloodhound", "metasploit", "wireshark",
            "tryhackme", "hackthebox", "auditor", "overnight", "helpdesk",
            "resume", "interview", "recruiter", "certification", "graveyard",
            "nightshift", "audit", "hospitality", "marriott", "hyatt", "hilton",
        }
        return {w for w in words if w not in _NOT_A_NAME}

    # -- index --------------------------------------------------------------
    def _roots(self) -> list[Path]:
        """Everything worth remembering across, not just HIVE's own folder.

        VOSS and HackBot keep their own versioned memory logs outside the vault,
        and those logs are the record of most of the work — copying them in would
        create two truths that drift, so they are indexed where they live.
        Absolute paths are taken as given; bare names are vault-relative.
        """
        roots = [self.ltm.root]
        extra = self.cfg.get("memory.associations.extra_folders", []) or []
        for name in extra:
            raw = Path(str(name)).expanduser()
            p = raw if raw.is_absolute() else self.cfg.vault / raw
            if p.is_dir():
                roots.append(p)
        return roots

    def _files(self) -> list[Path]:
        cap = int(self.cfg.get("memory.associations.max_files", 800))
        out: list[Path] = []
        for root in self._roots():
            if not root.is_dir():
                continue
            for path in root.rglob("*.md"):
                if "_state" in path.parts:
                    continue
                out.append(path)
                if len(out) >= cap:
                    return out
        return out

    def build_async(self) -> None:
        if not self.enabled:
            self._ready.set()
            return
        threading.Thread(target=self._build_guarded, name="hive-assoc",
                         daemon=True).start()

    def _build_guarded(self) -> None:
        try:
            self.build()
        except Exception as exc:                       # never break a session
            self.bus.log(f"association index failed: {exc}", "warn")
            self._ready.set()

    def build(self) -> None:
        """(Re)build the index, reusing the cache for files that have not changed.

        The cache is keyed on mtime per file and holds each file's extracted
        mentions, so a session that touched one note re-parses one note.
        """
        t0 = time.monotonic()
        cached = self.ltm.load_state(STATE_KEY, {}).get("files", {})
        fresh: dict[str, dict] = {}
        parsed = 0

        for path in self._files():
            key = str(path)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            prior = cached.get(key)
            if prior and abs(float(prior.get("mtime", 0.0)) - mtime) < 0.5:
                fresh[key] = prior
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            fresh[key] = {"mtime": mtime, "hits": self._parse(path, text)}
            parsed += 1

        index: dict[str, list[Mention]] = {}
        display: dict[str, str] = {}
        for entry in fresh.values():
            for hit in entry.get("hits", []):
                # [key, display, date, source, excerpt, verb, closed]
                try:
                    k, disp, when, source, excerpt, verb, closed = hit
                except (ValueError, TypeError):
                    continue
                index.setdefault(k, []).append(
                    Mention(when, source, excerpt, verb, bool(closed)))
                # Prefer the best-cased form seen anywhere. The same entity
                # arrives from two passes — "eJPT" from the vocabulary and
                # "Kerberoasting" from the proper-noun scan — and which one lands
                # first is an accident of line order, while the casing decides
                # whether the entity may be spoken aloud at all.
                prior = display.get(k)
                if prior is None or (name_like(disp) and not name_like(prior)):
                    display[k] = disp

        # Newest first, and bounded: an entity with forty sightings tells us
        # nothing the most recent eight do not.
        for k, mentions in index.items():
            mentions.sort(key=lambda m: m.date, reverse=True)
            del mentions[8:]

        with self._lock:
            self._entities, self._display = index, display
        self.ltm.save_state(STATE_KEY, {"files": fresh})
        self._ready.set()
        self.bus.log(
            f"associations: {len(index)} entities, {parsed} notes reparsed "
            f"({(time.monotonic() - t0) * 1000:.0f}ms)", "memory")

    # -- parsing ------------------------------------------------------------
    def _file_date(self, path: Path, text: str) -> str:
        m = re.search(r"(?m)^date:\s*(\d{4}-\d{2}-\d{2})", text)
        if m:
            return m.group(1)
        # 00_Sessions/YYYY-MM-DD/HHMM Session.md
        if _LINE_DATE.fullmatch(path.parent.name):
            return path.parent.name
        # 20_Inbox/YYYY-MM-DD.md
        if _LINE_DATE.fullmatch(path.stem):
            return path.stem
        m = _TD_NAME.match(path.stem)
        if m:
            mm, dd, yy = m.groups()
            return f"20{yy}-{mm}-{dd}"
        try:
            return _iso(date.fromtimestamp(path.stat().st_mtime))
        except OSError:
            return _iso(date.today())

    def _parse(self, path: Path, text: str) -> list[list]:
        base = self._file_date(path, text)
        source = path.stem
        hits: list[list] = []
        seen_on_line: set[str] = set()

        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("---", "```", "|", "tags:", "#")):
                continue
            m = _LINE_DATE.search(line)
            when = m.group(0) if m else base
            verb = ""
            for pat, label in _LOOP_VERBS:
                if pat.search(line):
                    verb = label
                    break
            closed = bool(_RESOLVED.search(line))
            excerpt = re.sub(r"\s+", " ", line)[:200]

            seen_on_line.clear()
            for name in self._entities_in(line):
                key = normalise(name)
                if key in seen_on_line:
                    continue
                seen_on_line.add(key)
                hits.append([key, name, when, source, excerpt, verb, closed])
        return hits

    def _entities_in(self, line: str, prose: bool = True) -> list[str]:
        """Names and known technical terms on one line of text.

        `prose` guards the sentence-start heuristic, and it is on for indexing and
        off for lookup. Indexing reads whole notes, where a capitalised word in
        position zero is usually "Reviewed" or "Applied" starting a bullet, and
        letting those in fills the index with verbs. Lookup reads an utterance,
        where position zero is where the subject actually is — "Kerberoasting is
        the one I keep hitting" — and a false positive there costs nothing at all,
        because a name that was never indexed simply returns no mentions.
        """
        out: list[str] = []
        # Strip the scaffolding LongTermMemory writes, so "JAY" and "HIVE" from a
        # transcript line do not become entities and the speaker's words do.
        body = re.sub(r"^\W*\**\d{1,2}:\d{2}(:\d{2})?\**\W*", "", line)
        body = re.sub(r"^(JAY|HIVE)\s+", "", body)
        # ...and the bolded field labels in the operator notes ("**Name:** Jay"),
        # which are template furniture rather than anything Jay wrote.
        body = re.sub(r"^\**[A-Z][a-z]+:\**\s*", "", body)

        for m in _PROPER.finditer(body):
            name = m.group(1).strip(" .'’-")
            if len(name) < 4:
                continue
            words = name.split()
            lowered = [w.lower().strip(".'’-") for w in words]
            if all(w in _NOT_A_NAME for w in lowered):
                continue
            # A single capitalised word at the very start of the line is usually
            # a sentence, not a name — unless it is vocabulary we already know.
            if (prose and len(words) == 1 and m.start() == 0
                    and lowered[0] not in self._vocab):
                continue
            if lowered[0] in _NOT_A_NAME and len(words) == 1:
                continue
            out.append(name)

        for term in self._vocab:
            if len(term) <= 3:
                continue
            # Match case-insensitively but keep what was actually written, so
            # "eJPT" is remembered as eJPT rather than as the lowercase key.
            m = re.search(rf"\b{re.escape(term)}\b", body, re.I)
            if m:
                out.append(m.group(0))
        return out

    # -- query --------------------------------------------------------------
    def entities_in(self, text: str) -> list[str]:
        """Public: the names this text mentions. Used to seed follow-up questions."""
        return self._entities_in(text or "", prose=False)

    def best_form(self, name: str) -> str:
        """The canonical spelling of a name.

        Whisper transcribes "tryhackme" and "ejpt"; the notes say "TryHackMe" and
        "eJPT". Questions built from a transcript get echoed into the terminal
        verbatim, so it is worth a dict lookup to spell them properly. The fixed
        table is consulted before the index because the index is built in the
        background and the first turn of a session can beat it.
        """
        key = normalise(name)
        fixed = _CANONICAL.get(key)
        if fixed:
            return fixed
        with self._lock:
            return self._display.get(key, name)

    def _open_loop(self, mentions: list[Mention]) -> Mention | None:
        """The newest commitment with no later resolution of the same entity."""
        loop = next((m for m in mentions if m.verb and not m.closed), None)
        if loop is None:
            return None
        if any(m.closed and m.date >= loop.date for m in mentions):
            return None
        return loop

    def connect(self, text: str, limit: int = 3,
                exclude: set[str] | None = None,
                strict: bool = True) -> list[Connection]:
        """What in `text` the vault has seen before, strongest first.

        `strict` is the difference between volunteering and being asked. Left on,
        a match has to earn the interruption — seen more than once, or carrying an
        unfinished commitment, and not first seen today. Turned off, everything
        the index holds comes back, which is what the terminal's `recall` wants:
        the operator asked, so the answer is the whole answer.

        Two passes, and the second is the one that matters.

        DIRECT is the obvious half: a name in this turn that the vault already
        knows. It only counts for something seen more than once or carrying an
        unfinished commitment — "you mentioned that before" is not worth a
        sentence of anyone's time, and an assistant that says it every turn gets
        muted.

        BRIDGE is the half that makes this feel like memory rather than search.
        Asking about "junior analyst positions" names nothing at all: those
        are ordinary words. But they appear in a note that also says a Acme and
        a Globex were applied to, and *those* are what the question is really
        about. So a topic that matches nothing sayable is used to reach the names
        sitting beside it, and only ones with an open loop come back — otherwise
        every generic word would drag half the vault along with it.
        """
        if not self.enabled or not self._ready.is_set():
            return []
        exclude = exclude or set()
        found: list[Connection] = []

        with self._lock:
            index, display = self._entities, self._display
            if not index:
                return []
            seen: set[str] = set(exclude)
            bridges: list[str] = []

            for name in self._entities_in(text or "", prose=False):
                key = normalise(name)
                if key in seen or len(key) < 4:
                    continue
                mentions = index.get(key)
                if not mentions:
                    continue
                shown = display.get(key, name)
                if strict and not name_like(shown):
                    # Not sayable, but a good handle to pull on.
                    bridges.append(key)
                    continue
                seen.add(key)

                loop = self._open_loop(mentions)
                newest = mentions[0]
                age = days_since(newest.date)
                if strict and len(mentions) < 2 and loop is None:
                    continue
                # Everything in here happened today — that is the current session
                # talking to itself, not memory.
                if strict and age <= 0 and loop is None:
                    continue
                found.append(Connection(
                    display=shown, days_ago=age, times=len(mentions),
                    source=newest.source,
                    loop_verb=loop.verb if loop else "",
                    loop_days=days_since(loop.date) if loop else 0,
                    excerpt=(loop or newest).excerpt,
                ))

            for key in bridges:
                topic = display.get(key, key)
                for mention in index.get(key, [])[:4]:
                    for other in self._entities_in(mention.excerpt):
                        okey = normalise(other)
                        if okey == key or okey in seen or len(okey) < 4:
                            continue
                        omentions = index.get(okey)
                        if not omentions:
                            continue
                        oshown = display.get(okey, other)
                        if not name_like(oshown):
                            continue
                        loop = self._open_loop(omentions)
                        if loop is None:
                            continue
                        seen.add(okey)
                        found.append(Connection(
                            display=oshown, days_ago=days_since(omentions[0].date),
                            times=len(omentions), source=loop.source,
                            loop_verb=loop.verb, loop_days=days_since(loop.date),
                            excerpt=loop.excerpt, via=topic,
                        ))

        found = self._drop_subsumed(found)
        found.sort(key=lambda c: -c.strength)
        return found[:limit]

    @staticmethod
    def _drop_subsumed(found: list[Connection]) -> list[Connection]:
        """Keep the most specific form of each name.

        "Acme" and "Acme Regional" index separately and both match the same
        line, which produced "you applied to Acme Regional, Acme and Globex
        Marquis" — three places, two of which are one place. A name whose words
        are wholly contained in another's is the same thing said less precisely.
        """
        keep: list[Connection] = []
        for c in found:
            words = set(normalise(c.display).split())
            if any(words < set(normalise(o.display).split()) for o in found):
                continue
            if any(normalise(c.display) == normalise(o.display) for o in keep):
                continue
            keep.append(c)
        return keep

    def fact_line(self, connections: list[Connection]) -> str:
        """The same connections phrased for the model's context, not for speech."""
        parts = []
        for c in connections:
            bit = f"{c.display} (seen {c.times}x, last {human_when(c.days_ago)}"
            if c.open_loop:
                bit += f"; you {c.loop_verb} it {human_when(c.loop_days)} "
                bit += "with no recorded outcome"
            parts.append(bit + f", in note '{c.source}')")
        return " || ".join(parts)

    def summary(self, term: str) -> str:
        """Human-readable recall for the terminal's `recall` command."""
        if not self._ready.is_set():
            return "the association index is still building"
        cons = self.connect(term, limit=6, strict=False)
        if not cons:
            with self._lock:
                known = len(self._entities)
            return (f"nothing connected to {term!r} "
                    f"({known} entities indexed)")
        out = []
        for c in cons:
            head = (f"  {c.display}  ·  {c.times} mention(s), last "
                    f"{human_when(c.days_ago)}  ·  {c.source}")
            if c.open_loop:
                head += f"\n      open loop: you {c.loop_verb} it {human_when(c.loop_days)}"
            out.append(head + f"\n      {c.excerpt[:120]}")
        return "\n".join(out)

    @property
    def ready(self) -> bool:
        return self._ready.is_set()
