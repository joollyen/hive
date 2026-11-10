"""End-to-end self test: `hive --selftest`.

Exercises every layer without a GUI or a microphone, and reports the real latency
of each stage. Run this after changing anything — it is much faster than talking
to the thing and guessing why it feels slow.

The ASR check is a genuine round trip: HIVE synthesizes a sentence with its own
voice, then transcribes its own audio. That covers both engines, the resampling
between them, and whether CUDA is actually being used, in one measurement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from . import _cuda

OK, BAD, WARN = "\033[92m  OK \033[0m", "\033[91mFAIL \033[0m", "\033[93mWARN \033[0m"


def _t(label: str, fn):
    t0 = time.monotonic()
    try:
        result = fn()
        ms = (time.monotonic() - t0) * 1000
        return True, result, ms
    except Exception as exc:
        ms = (time.monotonic() - t0) * 1000
        return False, exc, ms


def run() -> int:  # noqa: C901
    _cuda.preload()
    failures = 0
    asr_ms: float | None = None
    print("\n\033[96mHIVE self test\033[0m\n" + "─" * 62)

    # -- config -------------------------------------------------------------
    from .config import load
    from .events import EventBus
    ok, cfg, ms = _t("config", load)
    if not ok:
        print(f"{BAD} config           {cfg}")
        return 1
    print(f"{OK} config           {ms:5.0f}ms   voice={cfg.get('tts.voice')} "
          f"asr={cfg.get('asr.model')}")
    bus = EventBus()

    # -- CUDA ---------------------------------------------------------------
    cuda = _cuda.cuda_available()
    print(f"{OK if cuda else WARN} cuda             "
          f"{'CUDAExecutionProvider present' if cuda else 'CPU only — TTS will be ~2x slower'}")

    # -- vault --------------------------------------------------------------
    from .memory.ltm import LongTermMemory
    ltm = LongTermMemory(cfg, bus)
    ok, res, ms = _t("vault", lambda: (ltm.ensure(), ltm.root)[1])
    if ok:
        print(f"{OK} vault            {ms:5.0f}ms   {res}")
    else:
        print(f"{BAD} vault            {res}")
        failures += 1

    # -- intent grammar -----------------------------------------------------
    from .cognition.intents import classify, detect_wake
    cases = [
        ("HIVE open up a new google docs window for me on google chrome", "launch"),
        ("HIVE are you there", "presence"),
        ("hive make a note that the eJPT exam is friday", "note_add"),
        ("hive what time is it", "time"),
        ("hive how much disk space do i have", "system_query"),
        ("hive good night", "shutdown"),
        # Verbatim from a real session where HIVE heard everything and answered
        # nothing. Whisper wrote the name as "Hi" and as the digit "5", and the
        # pause after it split the utterance in two.
        ("Hi, are you there?", "presence"),
        ("Are you there?", "presence"),
        ("5 open chrome", "launch"),
        ("high open obsidian", "launch"),
        # v5 regressions, all from real sessions.
        #
        # A launch verb aimed at a subject rather than an application. This one
        # opened Seahorse: "launch" matched, the whole sentence went to the
        # resolver as an app name, and the fuzzy scan scored "Passwords and
        # Keys" at 85 against it.
        #
        # These expected "converse" until v6. The rule they guard has not
        # changed — a request naming a kind of *reading* must never reach the
        # launch resolver — but with a network there is now something better to
        # do with it than talk, so it resolves to `research` and the objective
        # side decides whether that means a search or a conversation. The
        # property is asserted directly in `never_launch` below, which is what
        # the original cases were really testing.
        ("hive pull up some university free source articles on smoother and "
         "more human sounding ai voice agents", "research"),
        ("hive find me research on emotional intelligence in ai", "research"),

        # -- v6: HIVE-Q ------------------------------------------------------
        # The override, in the phrasings someone actually reaches for. It sits
        # first in the grammar precisely so "stop the query" cannot be taken as
        # a request to stop the music.
        ("hive end query", "query_end"),
        ("hive stop the query", "query_end"),
        ("hive cancel the query", "query_end"),
        ("hive run the query", "query_run"),
        ("hive open hive q", "query_open"),

        # -- v6: the network -------------------------------------------------
        ("hive open youtube and type into the search bar "
         "'betterminecraft+ hardcore gameplay'", "site_search"),
        ("hive research options for me on best buy for keyboards similar to "
         "that of the vanguard corsair one", "site_search"),
        ("hive search the web for rtx 3070 ti undervolt", "web_search"),
        ("hive look up kerberoasting online", "web_search"),
        ("hive find me research articles on neural networks", "research"),
        # And the request that matched nothing at all, so HIVE said it was
        # creating the file, said it again, and never wrote one.
        ("hive create a markdown file with everything we talked about", "note_write"),
        ("hive write all of that up in obsidian", "note_write"),
        ("hive document everything we just discussed", "note_write"),
        # Neither of which may cost the paths they sit next to.
        ("hive pull up spotify", "launch"),
        ("hive open hack the box", "launch"),
        ("hive make a note that the exam is friday", "note_add"),
        # -- v8: the status briefing -----------------------------------------
        # Jay's phrasing, verbatim. It used to fall through to conversation,
        # where the model improvised a status out of nothing — which is the one
        # kind of answer this request must never get.
        ("hive, can you give me a current status update", "status_update"),
        ("hive give me a status update", "status_update"),
        ("hive status report", "status_update"),
        # Neither of the two rules it sits in front of may lose a phrasing to it.
        ("hive status check", "presence"),
        ("hive system status", "system_query"),

        # The name on its own wakes it and says nothing — the screen already
        # says LISTENING. An actual question still gets an actual answer.
        ("HIVE", "wake_only"),
        ("hive", "wake_only"),
        ("HIVE, are you there?", "presence"),
    ]
    bad = []
    for phrase, expect in cases:
        woke, rest = detect_wake(phrase, cfg)
        got = classify(rest if woke else phrase).name
        if got != expect or not woke:
            bad.append(f"{phrase[:52]!r} -> woke={woke} {got} (wanted {expect})")

    # And these must NOT wake it: Whisper's silence hallucinations, bare filler,
    # and ordinary speech aimed at someone else in the room.
    quiet = ["Thx for watching!", "Thanks for watching!", "There", "hi", "hey",
             "yo", "so anyway I told him no", "5", "okay then"]
    noisy = [p for p in quiet if detect_wake(p, cfg)[0]]

    # The property the article/paper rules exist for, asserted as itself rather
    # than through a proxy for it. Whatever these resolve to, it must never be
    # `launch` — that is the failure that opened Seahorse.
    never_launch = [
        "pull up some university free source articles on smoother and more "
        "human sounding ai voice agents",
        "find me research on emotional intelligence in ai",
        "pull up some articles on human sounding voice agents",
        "some papers about voice agent prosody",
    ]
    launched = [p for p in never_launch if classify(p).name == "launch"]

    # And the fidelity rule HIVE-Q depends on: a typed search string crosses the
    # grammar exactly as written. Normalisation deletes "+", so the site rules
    # parse the raw text — without that, a literal search loses characters
    # between the editor and the browser.
    q3 = classify("open youtube and type into the search bar "
                  "'betterminecraft+ hardcore gameplay'")
    verbatim = q3.slots.get("query") == "betterminecraft+ hardcore gameplay"

    if bad or noisy or launched or not verbatim:
        print(f"{BAD} intents          {len(bad)} wrong, {len(noisy)} false wakes")
        for b in bad:
            print(f"       {b}")
        for n in noisy:
            print(f"       false wake on {n!r}")
        for p in launched:
            print(f"       resolved to a LAUNCH: {p[:60]!r}")
        if not verbatim:
            print(f"       search string was mangled: "
                  f"{q3.slots.get('query')!r} — the '+' must survive")
        failures += 1
    else:
        print(f"{OK} intents          {len(cases)}/{len(cases)} correct, "
              f"0/{len(quiet)} false wakes, {len(never_launch)} subjects "
              f"kept off the launcher")

    # -- launcher -----------------------------------------------------------
    from .skills.launcher import Launcher
    L = Launcher(cfg, bus)
    checks = [("google docs", "url"), ("chrome", "app"), ("obsidian", "app"),
              ("claude", "url"), ("terminal", "app"), ("spotify", "app"),
              ("obsidien", "app"), ("hack the box", "url")]
    miss = [n for n, k in checks if (L.resolve(n) or None) is None]
    # The other half of the same check: a sentence is not an application name,
    # and the resolver must say so rather than finding the nearest thing.
    sentences = [
        "university free source articles on smoother and more human sounding "
        "ai voice agents",
        "some papers about voice agent prosody",
        "the difference between a comma and a semicolon",
    ]
    wrong = [(s, r.label) for s in sentences if (r := L.resolve(s)) is not None]
    if miss or wrong:
        if miss:
            print(f"{WARN} launcher         unresolved: {', '.join(miss)}")
        for s, label in wrong:
            print(f"{BAD} launcher         {s[:40]!r} resolved to {label!r}")
            failures += 1
    else:
        print(f"{OK} launcher         {len(checks)}/{len(checks)} resolved, "
              f"{len(sentences)}/{len(sentences)} sentences refused")

    # -- network -------------------------------------------------------------
    # Checked before the heavy engines, because if HIVE is offline the whole
    # research half of v6 falls back to conversation and that is worth knowing
    # at the top of the report rather than the bottom.
    from .net import Web, check_url, Blocked
    web = Web(cfg, bus)
    if not bool(cfg.get("net.enabled", True)):
        print(f"{WARN} network          disabled in config — research and search "
              f"fall back to conversation")
    elif not web.online(force=True):
        print(f"{WARN} network          unreachable — HIVE will say so rather "
              f"than pretend, and research falls back to conversation")
    else:
        from .net import Offline
        t0 = time.monotonic()
        try:
            hits = web.search("kerberoasting detection")
            search_ms = (time.monotonic() - t0) * 1000
        except Offline as exc:
            # A bot challenge is not a bug in HIVE. It is what a search
            # front-end does to an address that has just run a dozen queries,
            # and the point of this line is that it says so rather than
            # reporting an empty result set as if that were the answer.
            hits, search_ms = [], 0.0
            print(f"{WARN} web search       {exc}")
            print(f"       not a failure — HIVE says this out loud rather than "
                  f"reporting 'nothing found'. Try again in a few minutes.")
        except Exception as exc:
            hits, search_ms = [], 0.0
            print(f"{BAD} web search       {type(exc).__name__}: {exc}")
            failures += 1
        t0 = time.monotonic()
        try:
            papers = web.papers("neural networks", limit=3)
            arxiv_ms = (time.monotonic() - t0) * 1000
        except Exception as exc:
            papers, arxiv_ms = [], 0.0
            print(f"{WARN} arxiv            {type(exc).__name__}: {exc}")
        if hits:
            print(f"{OK} web search       {search_ms:5.0f}ms -> {len(hits)} results "
                  f"({hits[0].host})")
            # Sponsored rows are indistinguishable from results in the markup
            # and are never what was asked for.
            ads = [h for h in hits if "duckduckgo.com/y.js" in h.url]
            if ads:
                print(f"{BAD} web search       {len(ads)} sponsored row(s) leaked "
                      f"through the filter")
                failures += 1
        elif search_ms:
            print(f"{WARN} web search       returned nothing")
        if papers:
            print(f"{OK} arxiv            {arxiv_ms:5.0f}ms -> {len(papers)} papers "
                  f"({papers[0].meta.get('year', '?')})")

    # The address guard. A URL from a search result is attacker-influenced by
    # definition, and there is an unauthenticated Ollama on loopback.
    blocked = []
    for probe in ("http://127.0.0.1:11434/api/tags", "http://localhost/",
                  "http://192.168.1.1/", "file:///etc/passwd"):
        try:
            check_url(probe)
            blocked.append(probe)
        except Blocked:
            pass
        except Exception:
            pass          # DNS failure is a refusal too
    if blocked:
        print(f"{BAD} net guard        would have fetched: {', '.join(blocked)}")
        failures += 1
    else:
        print(f"{OK} net guard        loopback, private and non-http URLs refused")
    web.close()

    # -- HIVE-Q --------------------------------------------------------------
    # The queue, end to end, against a throwaway vault: parsing, the inactivity
    # watchdog, the override, and one markdown note per task including the
    # failures. Nothing here touches the microphone or the model.
    import shutil
    import tempfile
    import threading

    from .events import SPEAK_CHUNK
    from .query.runner import QueryRunner
    from .query.tasklist import DONE, INCOMPLETE, SKIPPED, parse

    tasks = parse("# a note, not a task\n"
                  "1. hive find me research articles on neural networks\n"
                  "2) hive open youtube and search for hardcore gameplay\n"
                  "\n"
                  "- hive check the disk \\\n     and say what is using it\n")
    parse_ok = (len(tasks) == 3
                and tasks[0].title.startswith("find me research")
                and tasks[2].text.endswith("using it"))

    real_vault = cfg.data["memory"]["vault_path"]
    tmp = tempfile.mkdtemp(prefix="hive-selftest-q-")
    qbad = []
    try:
        cfg.data["memory"]["vault_path"] = tmp
        for key, val in (("stall_after_s", 2.0), ("unwind_grace_s", 1.0),
                         ("gap_between_tasks_s", 0.05), ("announce_start", False),
                         ("announce_end", False), ("narrate_incomplete", False)):
            cfg.data["query"][key] = val

        class _Speaker:
            def __init__(self): self.stopped = threading.Event()
            def say(self, text, cached=True):
                bus.publish(SPEAK_CHUNK, text=text, first=True)
            def interrupt(self): self.stopped.set()

        class _Callosum:
            actions_log: list = []
            def handle(self, text, sw):
                if "STALL" in text:
                    time.sleep(30)              # silent: the watchdog must fire
                else:
                    bus.publish(SPEAK_CHUNK, text=f"answering {text[:24]}", first=True)
                # Built from the real dataclass rather than hand-rolled, so a
                # field added to Outcome cannot silently make this stub the
                # thing under test instead of the runner.
                from .cognition.callosum import Outcome
                return Outcome(spoke="ok", category="general")

        class _RT:
            _turn_lock = threading.Lock()
            speaker = _Speaker()
            callosum = _Callosum()
            class prefetch:
                @staticmethod
                def after_turn(*a, **k): pass
            ltm = LongTermMemory(cfg, bus)

        rt = _RT()
        rt.ltm.root = Path(tmp) / "HIVE"
        rt.ltm.ensure()
        runner = QueryRunner(cfg, bus, rt)
        runner.start("task one\ntask two STALL\ntask three\n")
        runner._thread.join(60)

        got = [t.status for t in runner.tasks]
        if got[0] != DONE:
            qbad.append(f"a clean task settled as {got[0]!r}, not done")
        if got[1] != INCOMPLETE:
            qbad.append(f"a silent task settled as {got[1]!r} — the "
                        f"{cfg.get('query.stall_after_s')}s watchdog did not fire")
        if not rt.speaker.stopped.is_set():
            qbad.append("the speaker was not interrupted when the task stalled")
        # A task that will not unwind must end the query rather than have the
        # next one started on top of it.
        if got[2] != SKIPPED:
            qbad.append(f"the query carried on past an unfinished task ({got[2]!r})")
        notes = sorted((Path(tmp) / "HIVE" / "query_tasks").glob("*.md"))
        if len(notes) != 2:
            qbad.append(f"{len(notes)} note(s) written, expected 2 "
                        f"(one per attempted task, pass or fail)")
        elif "Incomplete" not in notes[1].read_text(encoding="utf-8"):
            qbad.append("the failed task's note does not record that it failed")
    except Exception as exc:
        qbad.append(f"{type(exc).__name__}: {exc}")
    finally:
        cfg.data["memory"]["vault_path"] = real_vault
        shutil.rmtree(tmp, ignore_errors=True)

    if qbad or not parse_ok:
        print(f"{BAD} hive-q           {'; '.join(qbad) or 'buffer parsing wrong'}")
        failures += 1
    else:
        print(f"{OK} hive-q           3 tasks parsed, watchdog fired, override "
              f"honoured, 2 notes written")

    # -- prosody ------------------------------------------------------------
    # Kokoro reads a numeral as a word, so "1." is spoken as "one." and a
    # numbered list is heard as a countdown. And its vocabulary deletes the
    # hyphen, taking the pause it stood for with it. Both were audible in real
    # sessions; both are silent failures if this pass ever stops running.
    from .audio.prosody import shape, to_spoken
    listed = shape("The layers are: 1. Linguistic modelling of pitch and tempo. "
                   "2. Emotional intelligence - meaning sentiment analysis.")
    prosody_bad = []
    if "1." in listed or "2." in listed:
        prosody_bad.append("enumerators survive; they will be spoken as numbers")
    if " - " in listed or "—" not in listed:
        prosody_bad.append("hyphen not mapped to an em dash; the pause is deleted")
    if shape(listed) != listed:
        prosody_bad.append("not idempotent — the streaming path reshapes its own output")
    # v6: the dollar sign is outside Kokoro's vocabulary and is deleted silently,
    # so an unhandled price is spoken as a bare number with no unit — which is
    # the one form of a price that tells the listener nothing. Only reachable
    # since HIVE went online; a shopping search is the first thing in the system
    # that reads a price aloud.
    # `to_spoken`, not `shape` — the two halves of this module run at different
    # points. shape() restructures the text; to_spoken() rewrites what Kokoro
    # would mispronounce and runs per synthesis unit inside Speaker.synth.
    priced = to_spoken("The K70 is $129.99 and the K65 is $89.00, about 30% less.")
    if ("129 dollars 99" not in priced or "89 dollars" not in priced
            or "30 percent" not in priced):
        prosody_bad.append("prices or percentages lose their unit: " + priced[:90])
    if prosody_bad:
        print(f"{BAD} prosody          {'; '.join(prosody_bad)}")
        failures += 1
    else:
        print(f"{OK} prosody          enumeration, dashes and breath groups "
              f"({cfg.get('tts.prosody.breath_max_words')} words) OK")

    # -- status briefing ----------------------------------------------------
    # The briefing is a written message with authored numbers in it, so there is
    # no "is the count right" to check. What there is, and what this pass exists
    # for, is that every property which made the message as *written* undeliverable
    # stays fixed. All four were measured by phonemising through the same espeak
    # build the voice uses, and all four are silent — the briefing would still be
    # spoken, just wrongly.
    from . import status as status_mod
    from .audio.prosody import shape as shape_p
    from .audio.tts import split_units
    status_bad = []
    try:
        brief = status_mod.build(cfg)
        # 1. No path in anything spoken. espeak reads "/" aloud as the word
        #    "slash", so one path turns the closing line into "the live feed slash
        #    hive slash database slash livefeed slash".
        if "/" in brief.spoken:
            status_bad.append("a path or slash reached the spoken text")
        # 2. The date survives as one prosodic unit. An abbreviated month puts a
        #    full stop mid-sentence, and the splitter treats it as the end of one:
        #    "on Wednesday, September." [150ms, falling] "Twenty-third."
        units = [u.text for u in split_units(shape_p(brief.spoken),
                                             first_budget=140, rest_budget=420)]
        if any(u.rstrip().endswith(("Sept.", "Sep.", "Jan.", "Feb.", "Mar.",
                                    "Apr.", "Jun.", "Jul.", "Aug.", "Oct.",
                                    "Nov.", "Dec.")) for u in units):
            status_bad.append("an abbreviated month splits the date in two")
        # 3. Every unit is a closed sentence. A line without terminal punctuation
        #    is joined to the next one as a single unbroken breath — which is what
        #    the state reading did, running "Stable (recovered)" straight into
        #    "Since then, there have been...".
        unclosed = [u for u in units if u.strip() and u.strip()[-1] not in ".!?…"]
        if unclosed:
            status_bad.append(f"unterminated unit: {unclosed[0][:48]!r}")
        # 4. Number agreement, which no single template can carry: filled with a
        #    count of 1 the plural wording says "there have been 1 critical system
        #    failure that require your immediate attention", and a count of 0 says
        #    "there are also 0 low severity bugs I'd recommend reviewing".
        import copy as _copy
        from .config import Config as _Config

        def _render(**over) -> str:
            c = _Config.__new__(_Config)
            c.root, c.persona, c.apps = cfg.root, cfg.persona, cfg.apps
            c.calibrations, c._bags = cfg.calibrations, {}
            c.data = _copy.deepcopy(cfg.data)
            c.data.setdefault("status", {}).update(over)
            return status_mod.build(c).spoken

        one = _render(critical=1, low_severity=1)
        if "have been 1" in one or "are also 1" in one or " 1 bugs" in one:
            status_bad.append("singular count takes a plural verb: " + one[:80])
        zeroed = _render(critical=3, low_severity=0)
        if " 0 " in zeroed or "0 low" in zeroed:
            status_bad.append("a zero count states itself instead of dropping")
        empty = _render(critical=0, low_severity=0)
        if " 0 " in empty or "error log" in empty:
            status_bad.append("nothing-to-report still points at the error log")
        # The grammar has to reach it, including through the phrasing Jay uses.
        for utt in ("hive, can you give me a current status update",
                    "hive status report", "hive where do things stand"):
            if classify(utt).name != "status_update":
                status_bad.append(f"grammar misses {utt!r}")
        # And must not have taken these from the two rules it sits in front of.
        if classify("hive status check").name != "presence":
            status_bad.append("status_update swallowed 'status check'")
        if classify("hive system status").name != "system_query":
            status_bad.append("status_update swallowed 'system status'")
    except Exception as exc:
        status_bad.append(f"{type(exc).__name__}: {exc}")
    if status_bad:
        print(f"{BAD} status           {'; '.join(status_bad[:3])}")
        failures += 1
    else:
        print(f"{OK} status           {len(units)} closed units, no paths spoken, "
              f"agreement and grammar OK")

    # -- consolidation ------------------------------------------------------
    # The guarantee this checks is the one HIVE was worst at keeping. Seven
    # sessions in .logs/hive.log logged "consolidating session to vault" and
    # never logged anything again: the note was written *after* a 9-57s
    # heavy-model summary, so anything that killed the process during that
    # window took the conversation with it — and the conversation had been
    # complete in memory the whole time.
    #
    # Two properties, and a regression in either loses somebody's session:
    #   the transcript reaches disk before the model is asked for anything
    #   two writes produce one note, not two
    from .memory.ltm import SESSIONS
    from .runtime import Runtime

    real_vault = cfg.data["memory"]["vault_path"]
    vtmp = tempfile.mkdtemp(prefix="hive-selftest-vault-")
    vbad = []
    try:
        cfg.data["memory"]["vault_path"] = vtmp
        rt = Runtime(cfg, bus)
        rt.ltm.root = Path(vtmp) / "HIVE"
        rt.ltm.ensure()
        rt.stm.add_user("what is kerberoasting")
        rt.stm.add_hive("It abuses service principal names.")
        rt.callosum.actions_log.append("launch: Spotify")

        t0 = time.monotonic()
        name = rt.save_transcript()
        save_ms = (time.monotonic() - t0) * 1000
        note = rt.ltm.session_note

        if not name or note is None or not note.exists():
            vbad.append("the transcript was not written")
        else:
            body = note.read_text(encoding="utf-8")
            if "## Transcript" not in body or "kerberoasting" not in body.lower():
                vbad.append("the note has no transcript in it")
            # No model, no network, no lock — this is pure formatting over data
            # that already exists, and it is the whole safety margin.
            if save_ms > 500:
                vbad.append(f"the transcript save took {save_ms:.0f}ms; it must "
                            f"not be doing real work")

        # Second pass, as consolidate() makes: same note, summary replaced.
        rt.ltm.write_session(rt.stm, rt.callosum.actions_log,
                             "A summary written afterwards.", ["Kerberoasting"])
        notes = list((Path(vtmp) / "HIVE" / SESSIONS).rglob("*.md"))
        if len(notes) != 1:
            vbad.append(f"{len(notes)} notes written for one session, expected 1")
        else:
            after = notes[0].read_text(encoding="utf-8")
            if "A summary written afterwards." not in after:
                vbad.append("the summary pass did not land")
            if "kerberoasting" not in after.lower():
                vbad.append("the summary pass destroyed the transcript")
        if list((Path(vtmp) / "HIVE" / SESSIONS).rglob("*.part")):
            vbad.append("a half-written .part file was left behind")
        if not rt.save_now():
            vbad.append("`save` produced no receipt")
    except Exception as exc:
        vbad.append(f"{type(exc).__name__}: {exc}")
    finally:
        cfg.data["memory"]["vault_path"] = real_vault
        shutil.rmtree(vtmp, ignore_errors=True)

    if vbad:
        print(f"{BAD} consolidation    {'; '.join(vbad)}")
        failures += 1
    else:
        print(f"{OK} consolidation    transcript on disk in {save_ms:.0f}ms before "
              f"the model runs, one note per session, verified by read-back")

    # -- clipboard ----------------------------------------------------------
    # Pasted material is held as a quoted source rather than shoved through the
    # command line. The routing rule is the part worth guarding: a URL belongs
    # in the line and a paragraph does not, and getting that backwards makes
    # either pasting a path unusable or pasting an article useless.
    from .clip import Clippings

    clips = Clippings(cfg, bus)
    cbad = []
    inline = ["~/Documents/hackbot/report.md", "https://ex.com/" + "a" * 300,
              "grant voss"]
    blocks = ["one line\ntwo lines", " ".join(["word"] * 60)]
    for s in inline:
        if clips.is_block(s):
            cbad.append(f"{s[:24]!r} would be held instead of typed")
    for s in blocks:
        if not clips.is_block(s):
            cbad.append(f"{s[:24]!r} would be typed instead of held")

    clip = clips.hold("The AS-REQ is sent without pre-authentication. " * 4)
    if clip is None or len(clips) != 1:
        cbad.append("holding a clipping did not work")
    if clips.hold("The AS-REQ is sent without pre-authentication. " * 4) and len(clips) != 1:
        cbad.append("the same paste twice was held twice")
    block = clips.context_block()
    if "PASTED MATERIAL" not in block or "AS-REQ" not in block:
        cbad.append("the clipping does not reach the model's context")
    # Provenance is the point: an unattributed dump gets asserted back as
    # HIVE's own knowledge, which is how a forum post becomes something it said.
    if "quoted source" not in block:
        cbad.append("the context block does not say where the text came from")
    over = Clippings(cfg, bus)
    over.max_chars = 200
    big = over.hold("word " * 400)
    if big is None or not big.dropped:
        cbad.append("an oversized clipping was not truncated, or did not say so")
    elif "TRUNCATED" not in over.context_block():
        cbad.append("a truncated clipping does not tell the model it was cut")
    if not clips.clear().startswith("dropped"):
        cbad.append("clippings could not be cleared")

    if cbad:
        print(f"{BAD} clipboard        {'; '.join(cbad)}")
        failures += 1
    else:
        print(f"{OK} clipboard        paste routing (line vs source), dedupe, "
              f"truncation declared, provenance stated")

    # -- LLM ----------------------------------------------------------------
    from .cognition.llm import LLM
    llm = LLM(cfg, bus)
    if not llm.available():
        print(f"{BAD} llm              Ollama unreachable at {llm.host}")
        print("       start it with:  ollama serve")
        failures += 1
    else:
        msgs = [{"role": "system", "content": "Answer in exactly four words."},
                {"role": "user", "content": "Say hello to Jay."}]
        # Warm first. A cold Ollama load reports ~3s, which says nothing about
        # what a real turn costs — the runtime preloads this model at boot.
        t0 = time.monotonic()
        list(llm.stream(msgs))
        cold_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        first_ms, chunks = None, []
        for c in llm.stream(msgs):
            if first_ms is None:
                first_ms = (time.monotonic() - t0) * 1000
            chunks.append(c)
        total = (time.monotonic() - t0) * 1000
        print(f"{OK} llm fast         first={first_ms or -1:5.0f}ms total={total:5.0f}ms  "
              f"{cfg.get('llm.fast_model')}  (cold load was {cold_ms:.0f}ms)")
        print(f"       -> {''.join(chunks).strip()[:60]}")

    # -- TTS ----------------------------------------------------------------
    from .audio.tts import Speaker
    sp = Speaker(cfg, bus)
    done = []
    sp.preload(on_ready=lambda: done.append(1))
    if not sp.wait_ready(180):
        print(f"{BAD} tts              failed to load")
        return failures + 1
    t0 = time.monotonic()
    audio = sp.synth("Right here.", use_cache=False)
    tts_ms = (time.monotonic() - t0) * 1000
    dur = len(audio) / 24000 if len(audio) else 0
    if dur <= 0:
        print(f"{BAD} tts              produced no audio")
        failures += 1
    else:
        print(f"{OK} tts              {tts_ms:5.0f}ms -> {dur:.2f}s audio "
              f"(RTF {tts_ms/1000/dur:.3f}) on {sp.provider}")

    # -- VAD / endpointer ---------------------------------------------------
    # Regression guard. The v5 model needs 64 samples of context prepended to
    # every frame; without it the graph accepts the input silently and returns
    # ~0.002 for loud speech, so the endpointer never fires and HIVE simply never
    # answers. Nothing else in the test suite exercises this path.
    import queue as _queue

    from .audio.capture import AudioFrame
    from .audio.vad import CHUNK, Endpointer, SileroVAD

    if dur > 0:
        probe = sp.synth("HIVE, are you there?", use_cache=False)
        pa = np.asarray(probe, dtype=np.float32)
        pn = int(len(pa) * 16000 / 24000)
        p16 = np.interp(np.linspace(0, len(pa) - 1, pn), np.arange(len(pa)), pa).astype(np.float32)

        vad = SileroVAD(str(cfg.models / "vad" / "silero_vad.onnx"))
        probs = np.array([vad(p16[i:i + CHUNK]) for i in range(0, len(p16) - CHUNK, CHUNK)])
        thr = float(cfg.get("vad.threshold", 0.55))
        hot = int((probs >= thr).sum())

        class _Frames:
            def __init__(self, sig):
                self.frames: _queue.Queue = _queue.Queue(maxsize=4096)
                for i in range(0, len(sig) - CHUNK, CHUNK):
                    self.frames.put(AudioFrame(sig[i:i + CHUNK].copy(), i // CHUNK))

        # Trailing silence has to outlast the configured hangover, or the
        # endpointer is being asked to declare the utterance over before it has
        # been given the pause that ends it. A fixed one second was enough while
        # that was 380ms and quietly became a failing test the moment it wasn't.
        quiet_s = float(cfg.get("vad.min_silence_ms", 3000)) / 1000.0 * 1.4 + 0.6
        pad = np.zeros(int(16000 * quiet_s), dtype=np.float32)
        fired: list = []
        ep = Endpointer(cfg, bus, _Frames(np.concatenate([pad, p16, pad])),
                        lambda u: fired.append(u))
        ep.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 12 + quiet_s and not fired:
            time.sleep(0.05)
        ep.stop()

        if fired and hot:
            print(f"{OK} vad endpointer   fired at {fired[0].duration:.2f}s  "
                  f"(peak p={probs.max():.2f}, {hot}/{len(probs)} frames voiced)")
        else:
            print(f"{BAD} vad endpointer   did NOT fire — peak p={probs.max():.3f} "
                  f"vs threshold {thr}")
            print("       HIVE will hear you and never answer. Check the 64-sample "
                  "context window in audio/vad.py.")
            failures += 1

    # -- ASR (round trip through TTS output) --------------------------------
    from .audio.asr import Recognizer
    rec = Recognizer(cfg, bus)
    rec.preload()
    if not rec.wait_ready(180):
        print(f"{BAD} asr              failed to load")
        failures += 1
    elif dur > 0:
        probe = sp.synth("HIVE, open up a new Claude chat for me.", use_cache=False)
        a = np.asarray(probe, dtype=np.float32)
        n = int(len(a) * 16000 / 24000)
        a16 = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
        rec.transcribe(a16)                       # warm
        t0 = time.monotonic()
        text = rec.transcribe(a16)
        asr_ms = (time.monotonic() - t0) * 1000
        woke, rest = detect_wake(text, cfg)
        intent = classify(rest if woke else text)
        good = woke and intent.name == "launch"
        print(f"{OK if good else WARN} asr round trip   {asr_ms:5.0f}ms for "
              f"{len(a16)/16000:.2f}s audio")
        print(f"       heard: {text!r}")
        print(f"       -> wake={woke} intent={intent.name} "
              f"target={intent.slots.get('target')!r}")

    # -- barge-in -----------------------------------------------------------
    # "HIVE, stop" spoken over the top of HIVE talking. Two failures to guard
    # against and they pull in opposite directions: not hearing the operator at
    # all (which is what happened before v5, because the microphone was switched
    # off for the whole time HIVE was speaking), and hearing HIVE's own voice
    # well enough to obey a stop word in it. Simulated acoustically — HIVE at
    # speaker-bleed level, the interruption at close-mic level.
    from .cognition.intents import is_interrupt

    if dur > 0 and bool(cfg.get("audio.input.barge_in", True)):
        said = ("Start with authentication and injection flaws, then move on to "
                "the lab work you already have set up.")

        def _mic(text: str, gain: float) -> np.ndarray:
            a = np.asarray(sp.render_offline(text), dtype=np.float32) * gain
            n = int(len(a) * 16000 / 24000)
            return np.interp(np.linspace(0, len(a) - 1, n),
                             np.arange(len(a)), a).astype(np.float32)

        echo = _mic(said, 0.18)
        over = _mic("HIVE, stop.", 0.85)
        mixed = echo.copy()
        at = int(len(echo) * 0.45)
        n = min(len(over), len(mixed) - at)
        if n > 0:
            mixed[at:at + n] += over[:n]

        class _Talking:
            is_speaking = True

            def __init__(self, sig):
                self.frames: _queue.Queue = _queue.Queue(maxsize=8192)
                for i in range(0, len(sig) - CHUNK, CHUNK):
                    self.frames.put(AudioFrame(sig[i:i + CHUNK].copy(), i // CHUNK))

        def _heard_stop(sig) -> tuple[bool, int]:
            got: list = []
            ep2 = Endpointer(cfg, bus, _Talking(sig), lambda u: got.append(u))
            ep2.start()
            t1 = time.monotonic()
            while time.monotonic() - t1 < 20 and not got:
                time.sleep(0.05)
            time.sleep(0.9)
            ep2.stop()
            for u in got:
                txt = rec.transcribe(u.audio) if rec.ready else ""
                _w, rest = detect_wake(txt, cfg)
                if txt and is_interrupt(rest or txt):
                    return True, len(got)
            return False, len(got)

        self_stop, n_self = _heard_stop(echo)
        real_stop, n_real = _heard_stop(mixed)
        if self_stop or not real_stop:
            print(f"{BAD} barge-in         "
                  + ("HIVE interrupts itself" if self_stop
                     else "'HIVE, stop' over the top was not heard"))
            failures += 1
        else:
            print(f"{OK} barge-in         stop heard over HIVE's own voice "
                  f"({n_real} slice(s)); {n_self} from HIVE alone, none acted on")

    # -- audio devices ------------------------------------------------------
    try:
        import sounddevice as sd
        din = sd.query_devices(kind="input")
        dout = sd.query_devices(kind="output")
        print(f"{OK} audio in         {din['name']}")
        print(f"{OK} audio out        {dout['name']}")
    except Exception as exc:
        print(f"{WARN} audio devices    {exc}")

    # -- budget -------------------------------------------------------------
    print("─" * 62)
    # Reported as two numbers, because they are two different things and adding
    # them together hides the one that can actually be optimised. The hangover
    # is a deliberate wait — how long HIVE lets you pause before deciding you
    # have finished — and it is set in the config, not earned. The pipeline is
    # everything after that, and it is the only half the budget is about.
    hangover = float(cfg.get("vad.min_silence_ms", 3000))
    pipeline = (asr_ms if asr_ms is not None else 120.0) + 90.0
    budget = float(cfg.get("latency.target_first_audio_ms", 800))
    print(f"  pipeline (cached reply): ~{pipeline:.0f}ms to first audio"
          f"   {'within' if pipeline <= budget else 'OVER'} the {budget:.0f}ms budget")
    print(f"  plus a deliberate {hangover / 1000:.1f}s wait after you stop speaking "
          f"(vad.min_silence_ms)\n")

    if failures:
        print(f"\033[91m{failures} check(s) failed\033[0m\n")
    else:
        print("\033[92mall checks passed\033[0m\n")
    sp.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
