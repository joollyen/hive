"""The HIVE terminal — typed input alongside the voice channel.

Some things are miserable to say out loud and trivial to type: exact paths,
folder names with underscores, anything you need to be sure was heard correctly.
Granting filesystem access is the clearest case, which is why grants can only be
made here and never by voice — a misheard path should not be able to reach a file
operation.

The two channels share one session. A grant typed here becomes a grounded fact
the voice side can reason about immediately, so the intended flow works:

    terminal:  grant hyperintelligent_agent and voss
    voice:     "HIVE, organise the first folder the way the second one is."

Input is tried as a structured command, then as a natural-language grant, and
anything left over is handed to HIVE as an ordinary turn — so the terminal is
also just a quiet way to talk to it.

The clipboard commands are here rather than in the UI on purpose. `copy` and
`paste` need the transcript and the held clippings, both of which live on this
side, and putting them here means they work identically whether they were typed
or reached by a keystroke — the shortcuts in main.qml call straight into these.
"""

from __future__ import annotations

import re
import shlex

HELP = """HIVE terminal

  grant [--write] NAME [and NAME...]   allow access to a folder (read-only by default)
  revoke NAME | revoke all             withdraw access
  grants                               list what is currently granted
  find NAME                            search $HOME for a folder by that name
  tree NAME [DEPTH]                    show a granted folder's structure
  structure NAME                       summarise its shape (dirs, file types)
  mirror SRC like TEMPLATE             plan reorganising SRC to match TEMPLATE
  apply                                run the plan just shown
  voices                               list available voices
  voice SPEC                           switch voice, e.g. bf_lily:0.6 + bf_emma:0.4
  night [on|off|auto]                  night mode; no argument reports status
  jobs ROLE [in LOCATION]              open pre-filtered searches on the boards
  recall TERM                          what HIVE connects to TERM across sessions

  query                                open HIVE-Q, the task query
  query run                            start working down the query
  query status                         where it has got to
  end query                            STOP EVERYTHING — overrides any task in flight

  status                               the status briefing, as it would be spoken
  net                                  online or not, and what that changes
  search TERM                          search the web and read the results back
  calibration                          which lens is active, and time spent in each
  calibration why TEXT                 explain where that request would route

  copy                                 HIVE's last answer to the clipboard
  copy all | copy N                    the whole conversation, or the last N exchanges
  paste [QUESTION]                     hold the clipboard as a source, and ask about it
  clip                                 what is held right now
  clip drop | clip clear | clip show N let go of it, or read it back

  save                                 write the session to Obsidian now, and
                                       report the file it verified on disk

  clear                                clear this pane
  help                                 this

In the pane: drag to select lines, double-click to select a whole answer,
Ctrl+Shift+C copies it. Ctrl+V pastes — a block is held as a source rather than
crammed into the prompt; Ctrl+Shift+V forces it into the line verbatim.

Anything else is sent to HIVE as if you had said it out loud.
Grants can only be made here — never by voice."""

# "give access to folders titled X and Y" / "grant access to X, Y"
_NL_GRANT = re.compile(
    r"^(?:give|grant|allow)\s+(?:me\s+)?(?:access\s+)?(?:to\s+)?"
    r"(?:the\s+)?(?:folders?|directories|dirs?)?\s*(?:titled|named|called)?\s*(?P<rest>.+)$",
    re.I)
_NL_REVOKE = re.compile(r"^(?:revoke|remove|drop|forget)\s+(?:access\s+(?:to\s+)?)?(?P<rest>.+)$", re.I)
_MIRROR = re.compile(
    r"^(?:mirror|organi[sz]e|arrange|restructure)\s+(?P<src>.+?)\s+"
    r"(?:like|as|to match|the same (?:way )?as)\s+(?P<tpl>.+?)\s*$", re.I)


def _names(blob: str) -> list[str]:
    """Split "a, b and c" / "'x' and 'y'" into individual folder names."""
    blob = blob.strip().rstrip(".")
    parts = re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", blob)
    out = []
    for p in parts:
        p = p.strip().strip("\"'").strip()
        if p:
            out.append(p)
    return out


# The override. Matched before anything else is even tokenised, because this is
# the one command whose whole purpose is to work when the system is busy — and
# "busy" is exactly the state in which a typo, an unexpected quote or a shlex
# error would otherwise swallow it. Deliberately generous about phrasing: nobody
# reaches for this calmly.
_END_QUERY = re.compile(
    r"^\s*(?:end|stop|kill|abort|cancel|halt|quit|scrap)\s*"
    r"(?:the\s+|this\s+|my\s+)?(?:quer(?:y|ie)s?|queue|q)\s*!*\s*$", re.I)


class Terminal:
    def __init__(self, cfg, bus, workspace, speaker=None, on_request=None,
                 night=None, associations=None, calibrator=None, query=None,
                 web=None, stm=None, clippings=None, clipboard=None,
                 save_fn=None):
        self.cfg, self.bus = cfg, bus
        self.ws = workspace
        self.speaker = speaker
        self.night = night
        self.assoc = associations
        self.calib = calibrator
        self.query = query
        self.web = web
        self.stm = stm
        self.clippings = clippings
        # Injected by the UI once Qt exists — see ui/clipboard.py. None in
        # headless modes, which is the honest test for "there is no clipboard
        # here" rather than a guess at the platform.
        self.clipboard = clipboard
        # Runtime.save_now — writes the session note and returns a receipt read
        # back off disk. See _save_cmd.
        self.save_fn = save_fn
        self.on_request = on_request      # callable(text) -> spoken response
        self._plan: list = []

    # -- completion ---------------------------------------------------------
    # Command names plus whatever is granted right now. Rebuilt per keystroke,
    # which is fine: the grant map is a handful of entries and the alternative —
    # caching it — means a folder granted a second ago does not complete.
    COMMANDS = ("grant", "revoke", "grants", "find", "tree", "structure", "mirror",
                "apply", "voices", "voice", "night", "jobs", "recall",
                "calibration", "clear", "help", "--write",
                "query", "end query", "run", "status", "net", "search",
                "copy", "copy all", "paste", "clip", "clip drop", "clip clear",
                "save")

    def vocabulary(self) -> list[str]:
        words = list(self.COMMANDS)
        try:
            words += [g.path.name for g in self.ws.grants.values()]
        except Exception:
            pass
        return words

    # -- entry point --------------------------------------------------------
    def execute(self, line: str) -> str:
        line = (line or "").strip()
        if not line:
            return ""
        # Ahead of the try, ahead of tokenising, ahead of everything. `end query`
        # is the emergency brake and it must not be reachable only by way of a
        # code path that can raise. The bridge runs each submitted line on its
        # own thread, so this executes immediately even while a query task is
        # occupying the engine.
        if _END_QUERY.match(line):
            if self.query is None:
                return "HIVE-Q isn't running in this session."
            return self.query.end("you typed end query")
        try:
            return self._dispatch(line)
        except Exception as exc:                       # never kill the pane
            return f"error: {type(exc).__name__}: {exc}"

    def _dispatch(self, line: str) -> str:  # noqa: C901 - a command table
        low = line.lower()
        try:
            argv = shlex.split(line)
        except ValueError:
            argv = line.split()
        cmd = argv[0].lower() if argv else ""

        if cmd in ("help", "?", "h"):
            return HELP
        if cmd == "clear":
            return "\x00clear"
        if cmd in ("grants", "ls", "list"):
            return self.ws.summary()

        if cmd == "find":
            if len(argv) < 2:
                return "usage: find NAME"
            hits = self.ws.find(" ".join(argv[1:]))
            return ("\n".join(f"  {h}" for h in hits) if hits
                    else "no matching folder under your home directory")

        if cmd == "tree":
            if len(argv) < 2:
                return "usage: tree NAME [DEPTH]"
            depth = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else 3
            _ok, out = self.ws.tree(argv[1], depth)
            return out

        if cmd == "structure":
            if len(argv) < 2:
                return "usage: structure NAME"
            s = self.ws.structure(argv[1])
            if s is None:
                return f"no grant named {argv[1]!r}"
            exts = ", ".join(f"{k} x{v}" for k, v in s["extensions"].items())
            return (f"{s['name']}  ({'read+write' if s['writable'] else 'read-only'})\n"
                    f"  path       {s['path']}\n"
                    f"  directories {s['dir_count']}\n"
                    f"  files       {s['file_count']}\n"
                    f"  types       {exts or '(none)'}\n"
                    f"  layout      " + ", ".join(s["dirs"][:12] or ["(flat)"]))

        if cmd == "voices":
            if self.speaker is None:
                return "voice engine not available"
            vs = self.speaker.available_voices()
            brit = [v for v in vs if v.startswith(("bf_", "bm_"))]
            other = [v for v in vs if not v.startswith(("bf_", "bm_"))]
            return (f"current: {self.speaker.voice}\n"
                    f"british: {', '.join(brit)}\n"
                    f"other  : {', '.join(other[:14])}\n"
                    f"blend with weights, e.g.  voice bf_lily:0.6 + bf_emma:0.4")

        if cmd == "voice":
            if self.speaker is None or len(argv) < 2:
                return "usage: voice SPEC"
            spec = line[len(argv[0]):].strip()
            return (f"voice set to {spec}" if self.speaker.set_voice(spec)
                    else f"unknown voice: {spec}")

        if cmd == "night":
            if self.night is None:
                return "night mode unavailable"
            if len(argv) < 2:
                return self.night.status_line()
            arg = argv[1].lower()
            if arg in ("auto", "schedule", "default"):
                return self.night.set_forced(None)
            return self.night.set_forced(arg in ("on", "1", "true", "yes"))

        if cmd == "jobs":
            from .skills import jobs as jobskill
            rest = " ".join(argv[1:])
            if not rest:
                return "usage: jobs ROLE [in LOCATION]"
            m = re.search(r"^(?P<role>.+?)\s+(?:in|near)\s+(?P<loc>.+)$", rest, re.I)
            role = (m.group("role") if m else rest).strip()
            loc = (m.group("loc").strip() if m else "")
            targets = jobskill.build_urls(role, loc)
            lines = [f"  {t.label:14s} {t.url}" for t in targets]
            return ("Pre-filtered searches (open them with: "
                    f"HIVE, check indeed and ziprecruiter for '{role}'"
                    f"{' in ' + loc if loc else ''}):\n" + "\n".join(lines) +
                    "\n\nHIVE cannot read the listings — no network access. "
                    "Paste one in and it will compare it against your resume.")

        if cmd in ("calibration", "calib"):
            # The audit window onto silent routing. `calibration why <text>`
            # prints the score for every profile, so a turn that went somewhere
            # surprising can be explained rather than guessed at.
            if self.calib is None:
                return "calibration is not available"
            if len(argv) > 1 and argv[1].lower() == "why":
                rest = line.split(None, 2)
                if len(rest) < 3:
                    return "usage: calibration why SOME TEXT"
                return self.calib.explain(rest[2])
            return self.calib.summary()

        if cmd == "recall":
            # The inspection window onto associative memory. Everything HIVE may
            # volunteer unprompted can be checked here first, deliberately — a
            # recall you cannot audit is a recall you cannot trust.
            if self.assoc is None:
                return "associative recall is not available"
            if len(argv) < 2:
                return "usage: recall TERM"
            return self.assoc.summary(" ".join(argv[1:]))

        if cmd in ("query", "q", "hive-q", "hiveq"):
            return self._query_cmd(argv[1:])

        if cmd == "status":
            # The same inspection window `recall` gives associative memory, for
            # the same reason: a briefing that asserts two failures and four bugs
            # should be checkable without having to ask for it out loud and
            # listen. Prints what would be spoken, then the paths that are
            # deliberately not.
            from . import status as status_mod
            if not self.cfg.get("status.enabled", True):
                return ("the status briefing is switched off "
                        "(status.enabled in config/hive.toml)")
            b = status_mod.build(self.cfg)
            out = b.spoken
            if b.paths:
                out += "\n\n" + b.terminal_text()
            return out

        if cmd == "net":
            if self.web is None:
                return "the network layer isn't loaded"
            up = self.web.online(force="--force" in argv or "-f" in argv)
            if not up:
                return ("offline. Research and search fall back to conversation, "
                        "and HIVE will say so rather than pretend.")
            return ("online. HIVE can search the web, look papers up on arXiv, "
                    "fetch a page and read it, and open a site's search "
                    "pre-filled.\nRead-only: no logins, no forms, no posting. "
                    "The model and the vault never leave this machine.")

        if cmd == "search":
            if self.web is None or len(argv) < 2:
                return "usage: search TERM"
            term = line.split(None, 1)[1]
            try:
                results = self.web.search(term)
            except Exception as exc:
                return f"search failed: {exc}"
            if not results:
                return f"nothing came back for {term!r}"
            out = [f"{len(results)} result(s) for {term!r}:"]
            for r in results:
                out.append(f"  {r.title[:74]}")
                out.append(f"    {r.url}")
                if r.snippet:
                    out.append(f"    {r.snippet[:110]}")
            return "\n".join(out)

        if cmd in ("save", "consolidate", "vault"):
            return self._save_cmd()

        # -- clipboard --------------------------------------------------------
        # Ahead of the natural-language grammars below, which is what stops a
        # pasted paragraph beginning "Give the attacker one valid username…"
        # from being read as a request to grant a folder by that name.
        if cmd == "copy":
            return self.copy_out(" ".join(argv[1:]).strip().lower() or "last")

        if cmd == "paste":
            return self._paste_cmd(line)

        if cmd in ("clip", "clips", "held"):
            return self._clip_cmd(argv[1:])

        if cmd == "apply":
            if not self._plan:
                return "nothing planned. run `mirror SRC like TEMPLATE` first"
            ok, msg = self.ws.apply_mirror(self._plan)
            self._plan = []
            return msg if ok else f"failed: {msg}"

        # -- grants ---------------------------------------------------------
        if cmd == "grant":
            write = "--write" in argv or "-w" in argv
            rest = [a for a in argv[1:] if a not in ("--write", "-w")]
            if not rest:
                return "usage: grant [--write] NAME [and NAME...]"
            return self._grant_many(_names(" ".join(rest)), write)

        if cmd == "revoke":
            rest = " ".join(argv[1:]).strip()
            if rest.lower() in ("all", "*", "everything"):
                return self.ws.revoke_all()
            if not rest:
                return "usage: revoke NAME | revoke all"
            return "\n".join(self.ws.revoke(n)[1] for n in _names(rest))

        # -- natural language -------------------------------------------------
        m = _MIRROR.match(line)
        if m:
            ok, out, plan = self.ws.plan_mirror(m.group("src").strip(),
                                                m.group("tpl").strip())
            self._plan = plan if ok else []
            return out

        m = _NL_GRANT.match(line)
        if m and not low.startswith(("give me a", "grant me a")):
            write = bool(re.search(r"\b(write|full|read.?write|modify)\b", low))
            names = _names(re.sub(r"\b(access|permission|to)\b", " ", m.group("rest")))
            if names:
                return self._grant_many(names, write)

        m = _NL_REVOKE.match(line)
        if m:
            names = _names(m.group("rest"))
            if names and all(self.ws.get(n) for n in names):
                return "\n".join(self.ws.revoke(n)[1] for n in names)

        # -- fall through: treat as a spoken request --------------------------
        if self.on_request is None:
            return f"unknown command: {argv[0]}   (try `help`)"
        reply = self.on_request(line)
        return f"HIVE: {reply}" if reply else "(no response)"

    # -- the vault -----------------------------------------------------------
    def _save_cmd(self) -> str:
        """`save` — put the conversation in Obsidian now, and prove it.

        The session note is written on the way out anyway. This exists because
        "it gets written on the way out" is a promise, and the standing
        complaint is that the only way to check a promise was to go and open the
        vault. Run this whenever and it says the path, the line count and the
        size — read back off the file, not from the call that wrote it.

        It writes to the same note the shutdown will, so running it ten times
        does not leave ten notes. The last one wins and the shutdown overwrites
        it with the summarised version.
        """
        if self.save_fn is None:
            return "the vault writer isn't available in this mode."
        try:
            receipt = self.save_fn()
        except Exception as exc:
            return f"the vault write FAILED: {type(exc).__name__}: {exc}"
        return receipt or "nothing has been said yet — there is no session to save."

    # -- clipboard -----------------------------------------------------------
    def transcript_text(self, exchanges: int = 0) -> str:
        """The conversation as text. 0 means all of it.

        Built from short-term memory rather than from the pane, and the two are
        not the same thing on purpose. The pane holds what was drawn — speech
        split into synthesis units, command receipts, the calibration line. STM
        holds what was *said*, whole, including every spoken turn that happened
        while the terminal was shut. Copying the transcript should give you the
        conversation, not a screenshot of a log.
        """
        if self.stm is None:
            return ""
        who = self.cfg.get("identity.operator", "You")
        turns = [t for t in self.stm.transcript() if t.text.strip()]
        if exchanges > 0:
            # An exchange is a user turn and everything HIVE said after it, so
            # count backwards from the end by user turns rather than by lines.
            seen, start = 0, 0
            for i in range(len(turns) - 1, -1, -1):
                if turns[i].role == "user":
                    seen += 1
                    if seen == exchanges:
                        start = i
                        break
            turns = turns[start:]
        return "\n\n".join(f"{who if t.role == 'user' else 'HIVE'}: {t.text}"
                           for t in turns)

    def last_answer(self) -> str:
        """The last thing HIVE said, whole."""
        if self.stm is None:
            return ""
        for turn in reversed(self.stm.transcript()):
            if turn.role != "user" and turn.text.strip():
                return turn.text.strip()
        return ""

    def copy_out(self, what: str = "last") -> str:
        """`copy`, and the Ctrl+Shift+C shortcut. One implementation for both."""
        if self.clipboard is None:
            return "there's no clipboard in this mode."
        if self.stm is None:
            return "the transcript isn't available."

        if what in ("", "last", "answer", "that", "reply"):
            text, label = self.last_answer(), "HIVE's last answer"
            if not text:
                return "HIVE hasn't said anything yet."
        elif what in ("all", "transcript", "session", "everything"):
            text, label = self.transcript_text(), "the whole conversation"
            if not text:
                return "nothing has been said yet."
        elif what.isdigit() and int(what) > 0:
            n = int(what)
            text = self.transcript_text(n)
            label = f"the last {n} exchange(s)"
            if not text:
                return "nothing has been said yet."
        else:
            return "usage: copy | copy all | copy N"

        self.clipboard.set(text)
        return f"copied {label} — {len(text.split())} words."

    def _paste_cmd(self, line: str) -> str:
        """`paste`, and `paste QUESTION` in one move.

        The keyboard-free door into the same feature, and the one that always
        works: whatever is going on with a compositor, a focus quirk or a
        keybinding you have rebound, typing four letters cannot miss.
        """
        if self.clipboard is None:
            return "there's no clipboard in this mode."
        if self.clippings is None:
            return "pasted material isn't available in this mode."

        text = self.clipboard.read()
        if not text.strip():
            # The primary selection is the other half of the Linux clipboard and
            # it is full remarkably often — anything highlighted in a browser is
            # already in it, with no Ctrl+C involved.
            text = self.clipboard.read(selection=True)
        if not text.strip():
            return "the clipboard is empty."

        clip = self.clippings.hold(text)
        if clip is None:
            return "the clipboard is empty."

        _head, _sep, question = line.partition(" ")
        question = question.strip()
        receipt = (f"held {clip.words} words from the clipboard — {clip.label}\n"
                   f"  {self.clippings.status()} Ask about it in plain language.")
        if not question:
            return receipt

        # A question came with the paste. Run it now rather than making him
        # press Enter twice for one thought. The receipt goes out ahead of the
        # answer, so the pane shows what was held before HIVE reasons over it.
        from .events import TERMINAL_OUTPUT

        self.bus.publish(TERMINAL_OUTPUT, text=receipt, kind="sys")
        if self.on_request is None:
            return ""
        reply = self.on_request(question)
        return f"HIVE: {reply}" if reply else "(no response)"

    def _clip_cmd(self, args: list[str]) -> str:
        if self.clippings is None:
            return "pasted material isn't available in this mode."
        sub = args[0].lower() if args else ""
        if sub in ("", "status", "ls", "list"):
            return self.clippings.summary()
        if sub in ("drop", "pop", "undo"):
            return self.clippings.drop()
        if sub in ("clear", "none", "all"):
            return self.clippings.clear()
        if sub in ("show", "print", "read"):
            which = args[1] if len(args) > 1 else "1"
            if not which.isdigit():
                return "usage: clip show N"
            return self.clippings.show(int(which))
        return "usage: clip [drop|clear|show N]"

    # -- HIVE-Q --------------------------------------------------------------
    def _query_cmd(self, args: list[str]) -> str:
        if self.query is None:
            return "HIVE-Q isn't running in this session."
        from .events import QUERY_WINDOW
        from .query.tasklist import DONE, INCOMPLETE, PENDING, RUNNING, SKIPPED

        sub = args[0].lower() if args else "open"

        if sub in ("open", "show", "up"):
            self.bus.publish(QUERY_WINDOW, open=True)
            n = len(self.query.buffer.strip().splitlines())
            return (f"HIVE-Q is up ({n} line(s) in the editor). "
                    f"Ctrl+Shift+Q toggles it.")
        if sub in ("close", "hide"):
            self.bus.publish(QUERY_WINDOW, open=False)
            return "HIVE-Q hidden. The query keeps running if one is."
        if sub in ("run", "start", "go"):
            ok, msg = self.query.start()
            return msg if ok else f"-- {msg}"
        if sub in ("end", "stop", "kill", "abort"):
            return self.query.end("you typed query stop")

        if sub in ("status", "stat", "ls", "list"):
            snap = self.query.snapshot()
            if not snap["tasks"]:
                return "the query is empty. Type your tasks into HIVE-Q, one per line."
            glyph = {DONE: "  ok  ", INCOMPLETE: "  --  ", SKIPPED: "  ··  ",
                     RUNNING: "  >>  ", PENDING: "      "}
            head = ("running" if snap["running"] else "idle")
            if snap["paused"]:
                head += f" (held for {snap['pause_reason']})"
            out = [f"query: {head} — {snap['index'] or 0}/{snap['total']}"]
            for t in snap["tasks"]:
                out.append(f"{glyph.get(t['status'], '      ')}"
                           f"{t['index']}. {t['title'][:56]}")
                if t["reason"]:
                    out.append(f"          {t['reason'][:70]}")
                if t["note"]:
                    out.append(f"          -> query_tasks/{t['note']}")
            return "\n".join(out)

        return ("usage: query [open|close|run|status]   ·   `end query` stops "
                "everything")

    def _grant_many(self, names: list[str], write: bool) -> str:
        if not names:
            return "nothing to grant"
        out = []
        for n in names:
            ok, msg = self.ws.grant(n, write)
            out.append(("  ok   " if ok else "  --   ") + msg)
        granted = [n for n in names if self.ws.get(n)]
        if granted:
            out.append(f"\nHIVE can now see: {', '.join(granted)}. "
                       f"You can refer to them by name out loud.")
        return "\n".join(out)
