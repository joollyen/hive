"""Objective hemisphere — structural, deterministic, and the only side that can veto.

This half does not generate language. It decides what is *true* and what is
*permitted*, and it produces three things:

  action   a concrete, executable plan, or nothing
  facts    ground truth to inject into the model's context
  veto     a hard block, with a reason

The safety property of the whole architecture is an asymmetry: the objective
hemisphere can always overrule the subjective one, and the subjective one can
never overrule the objective one. Style is negotiable; permission and fact are
not. That is enforced in callosum.py, but the authority originates here.

Three concrete mechanisms implement "prevents dangerous misinformation":

1. Permission gating. Every action carries an ALLOW / CONFIRM / DENY verdict
   derived from configuration, not from the model's judgement. A model that
   decides it would be helpful to run a shell command still cannot.

2. Fact injection. Anything checkable — the date, the time, what is installed,
   what the vault actually contains — is computed and handed to the model as
   context, so it never has to guess and cannot drift.

3. Claim verification. After generation, statements about checkable facts are
   re-checked against ground truth. A contradiction is vetoed before it is
   spoken. This is the last line, and it catches the case where the model
   ignores injected context and asserts something anyway.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from .. import status
from ..events import (CLIPBOARD_SET, QUERY_WINDOW, TERMINAL_OUTPUT,
                      TERMINAL_TOGGLE, WEB_RESULTS)
from ..memory.ltm import WRITEUPS
from ..skills import system as sysinfo
from .intents import Category, Intent


class Permission(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass
class ActionPlan:
    kind: str
    label: str                                  # spoken back during confirmation
    run: Callable[[], tuple[bool, str]]
    permission: Permission = Permission.CONFIRM
    reversible: bool = True
    detail: str = ""


@dataclass
class Refusal:
    """A request HIVE understood and cannot fulfil.

    Carries a reason and a concrete alternative, because those are what separate
    a useful "no" from a dead end. A refusal with neither reads as a malfunction;
    this is the difference between HIVE freezing on "play my Deep Focus playlist"
    and it saying what is actually in the way and what it can do instead.
    """

    reason: str                       # why not, in plain speech
    offer: str = ""                   # the alternative, phrased as a next step
    action: ActionPlan | None = None  # that alternative, if it is executable

    def render(self, cfg) -> str:
        lead = cfg.line("cannot.lead")
        if not self.offer:
            return cfg.line("cannot.bare")
        pivot = cfg.line("cannot.pivot")
        return f"{lead} {self.reason}. {pivot} {self.offer}"


@dataclass
class Verdict:
    grounded: bool = False
    action: ActionPlan | None = None
    facts: dict[str, str] = field(default_factory=dict)
    veto: str | None = None
    speak_bank: str | None = None               # persona bank for a cached reply
    speak_text: str | None = None               # exact text, bypassing the model
    category: str = "general"
    escalate: bool = False
    refusal: Refusal | None = None


# --- destructive command patterns -----------------------------------------
# Configuration sets `destructive = "deny"`, meaning no confirmation can unlock
# these. Matching is intentionally broad: a false positive costs one refusal, a
# false negative costs a filesystem.

_DESTRUCTIVE = [
    (re.compile(r"\brm\s+(-\w*\s+)*-\w*[rf]", re.I), "recursive delete"),
    (re.compile(r"\bmkfs\b|\bfdisk\b|\bparted\b", re.I), "disk formatting"),
    (re.compile(r"\bdd\s+.*\bof=/dev/", re.I), "raw device write"),
    (re.compile(r"\bshred\b|\bwipefs\b", re.I), "secure erase"),
    (re.compile(r">\s*/dev/(sd|nvme|hd)", re.I), "raw device write"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh", re.I), "pipe-to-shell"),
    (re.compile(r":\(\)\s*\{.*\}\s*;\s*:", re.S), "fork bomb"),
    (re.compile(r"\bchmod\s+(-R\s+)?777\s+/(\s|$)", re.I), "root permission change"),
    (re.compile(r"\bchown\s+-R\b.*\s/(\s|$)", re.I), "root ownership change"),
    (re.compile(r"\b(shutdown|reboot|poweroff|halt)\b", re.I), "power state change"),
    (re.compile(r"\bkill(all)?\s+-9\b", re.I), "forced process kill"),
    (re.compile(r"\bsudo\b", re.I), "privilege escalation"),
    (re.compile(r"/etc/(passwd|shadow|sudoers)", re.I), "credential file access"),
]


def classify_destructive(text: str) -> str | None:
    for pat, why in _DESTRUCTIVE:
        if pat.search(text):
            return why
    return None


class ObjectiveHemisphere:
    def __init__(self, cfg, bus, launcher, ltm, speaker=None, workspace=None,
                 night=None, llm=None, associations=None, web=None, query=None):
        self.cfg, self.bus = cfg, bus
        self.launcher, self.ltm = launcher, ltm
        self.speaker = speaker
        self.workspace = workspace
        self.night = night
        # The network, and the queue. Both optional: HIVE has to keep working
        # with neither, and every path below that touches them checks first.
        self.web = web
        self.query = query
        # Used only by the document writer. The objective hemisphere does not
        # generate language for anything it says out loud — this is prose bound
        # for a file, checked into the vault where Jay can edit it, which is a
        # different thing from an assertion spoken with confidence.
        self.llm = llm
        self.associations = associations
        # Rebound by the runtime to the callosum's live list, so a write-up can
        # record what was actually done during the conversation it documents.
        self.actions_log: list[str] = []
        self._writing: threading.Thread | None = None

    # -- permissions --------------------------------------------------------
    def permission_for(self, capability: str) -> Permission:
        raw = self.cfg.get(f"permissions.{capability}", "confirm")
        try:
            return Permission(raw)
        except ValueError:
            return Permission.CONFIRM

    def vault_write_permitted(self, path) -> Permission:
        """Writes are free inside the configured roots and gated everywhere else."""
        allowed = self.cfg.get("permissions.vault_roots.allowed", ["HIVE"])
        try:
            rel = path.relative_to(self.cfg.vault)
        except (ValueError, AttributeError):
            return self.permission_for("file_write_outside_vault")
        top = rel.parts[0] if rel.parts else ""
        if top in allowed:
            return self.permission_for("vault_write")
        return self.permission_for("file_write_outside_vault")

    # -- grounded facts -----------------------------------------------------
    def base_facts(self) -> dict[str, str]:
        """Truth the model must not have to guess at.

        The clock is the highest-value item by a distance: models hallucinate the
        current date with total confidence, and a voice assistant that is wrong
        about what day it is loses trust immediately and permanently.
        """
        now = datetime.now()
        # Whether there is a network *right now*, not whether one was configured.
        # This is the single most consequential fact in the whole dictionary
        # after the clock: HIVE spent five versions telling Jay it was an offline
        # service, and that sentence is now true only some of the time. A model
        # told the wrong one either refuses work it could do, or promises a
        # search that will never happen.
        connected = bool(self.web is not None and self.web.online())
        base = {
            "current_datetime": now.strftime("%A, %B %d, %Y at %I:%M %p").replace(" 0", " "),
            "current_date_iso": now.strftime("%Y-%m-%d"),
            "operator": self.cfg.get("identity.operator", "Jay"),
            "assistant": self.cfg.get("identity.name", "HIVE"),
            "environment": "Ubuntu 26.04, GNOME on Wayland, RTX 3070 Ti",
            "capabilities": (
                "launch applications and URLs, open files, read and write notes in "
                "the Obsidian vault's HIVE folder, report system status on request, "
                "and — when asked — write a full markdown document into the vault "
                "covering everything discussed, which is a real action that takes "
                "a couple of minutes and reports the filename back"
            ) + ((
                ". You are also online right now, and that is a real capability "
                "rather than a claim: you can run a web search, look papers up on "
                "arXiv, fetch a page and read it, and open a site's own search "
                "already filled in. So when Jay asks for reading on a subject, the "
                "correct answer is to go and find it. Never say you are offline, "
                "never say you have no network access, and never say sources "
                "cannot be retrieved — all three were true until recently and none "
                "of them is true now"
            ) if connected else (
                ". Right now there is no network — the machine is offline, so "
                "searching, fetching a page and looking anything up are all "
                "genuinely unavailable this minute. Say that plainly if asked; "
                "do not invent a command that would do it"
            )),
            "limitations": (
                "cannot focus, move or resize existing windows (Wayland restriction); "
                "cannot log in to anything, fill in a form, post, buy or send; "
                "reads the web but never writes to it. The language model, the "
                "vault and every note stay on this machine — nothing said in a "
                "conversation is ever sent anywhere"
                if connected else
                "cannot focus, move or resize existing windows (Wayland restriction); "
                "cannot browse the web; has no network access at the moment. "
                "There is no way to fetch an article, a paper or a search result — "
                "so when asked for reading on a subject, answer from what you know "
                "and say plainly that the sources cannot be retrieved right now, "
                "rather than inventing a command that would do it"
            ),
            "network": "online" if connected else "offline",
        }
        if self.night is not None and self.night.active:
            base["mode"] = ("Night Mode: Jay is typing rather than speaking. Keep "
                            "answers short, say what you are doing, stay quiet in tone.")
        if self.workspace is not None and self.workspace.grants:
            base["granted_folders"] = ", ".join(
                f"{g.path.name} ({'read+write' if g.write else 'read-only'}) at {g.path}"
                for g in self.workspace.grants.values())
        return base

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, intent: Intent, stm) -> Verdict:  # noqa: C901 - dispatch table
        facts = self.base_facts()
        cat = intent.category.value

        if intent.name in ("presence", "presence_bare"):
            import time as _t
            repeat = (_t.time() - stm.last_presence_ts) < 60.0
            stm.last_presence_ts = _t.time()
            return Verdict(True, None, facts,
                           speak_bank="presence.repeat" if repeat else None,
                           speak_text=self.cfg.presence_line(repeat=repeat),
                           category="presence")

        if intent.name == "time":
            now = datetime.now()
            spoken = now.strftime("It's %I:%M %p on %A, %B %d").replace(" 0", " ")
            return Verdict(True, None, facts, speak_text=spoken, category="system")

        if intent.name == "launch":
            return self._plan_launch(intent, facts)

        if intent.name == "launch_ask":
            return self._plan_launch_ask(intent, facts)

        if intent.name == "job_search":
            return self._plan_jobs(intent, facts)

        if intent.name in ("research", "web_search"):
            return self._plan_lookup(intent, facts)

        if intent.name == "site_search":
            return self._plan_site_search(intent, facts)

        if intent.name.startswith("query_"):
            return self._plan_query(intent, facts)

        if intent.name == "note_write":
            return self._plan_document(intent, facts, stm)

        if intent.name == "note_add":
            return self._plan_note(intent, facts, stm)

        if intent.name == "note_recall":
            return self._plan_recall(intent, facts)

        if intent.name == "system_query":
            return self._plan_system(intent, facts)

        if intent.name == "status_update":
            return self._plan_status(facts)

        if intent.name == "voice_change":
            return self._plan_voice(intent, facts)

        if intent.name.startswith("media_"):
            return self._plan_media(intent, facts)

        if intent.name.startswith("workspace_"):
            return self._plan_workspace(intent, facts)

        if intent.name == "night_mode":
            if self.night is None:
                return Verdict(True, None, facts, speak_bank="veto.error",
                               category="system")
            # slots["want"] may legitimately be None (return to schedule), so
            # check presence rather than truthiness.
            want = intent.slots.get("want", True) if "want" in intent.slots else True
            said = self.night.set_forced(want)
            return Verdict(True, None, facts, speak_text=said, category="system")

        if intent.name in ("terminal_open", "terminal_close"):
            opening = intent.name == "terminal_open"

            def toggle(o=opening) -> tuple[bool, str]:
                self.bus.publish(TERMINAL_TOGGLE, open=o)
                return True, "open" if o else "closed"

            return Verdict(True, ActionPlan(
                kind="terminal", label="the HIVE terminal", run=toggle,
                permission=Permission.ALLOW), facts,
                speak_text="Terminal's up. Type away." if opening else "Closed.",
                category="system")

        if intent.name == "voice_list":
            voices = self.speaker.available_voices() if self.speaker else []
            british = [v for v in voices if v.startswith(("bf_", "bm_"))][:6]
            american = [v for v in voices if v.startswith(("af_", "am_"))][:6]
            txt = (f"I have {len(voices)} voices. British: {', '.join(british)}. "
                   f"American: {', '.join(american)}. Say switch to, then a name.")
            return Verdict(True, None, facts, speak_text=txt, category="system")

        # Free-form. Not grounded to an action, but still fact-supplied and
        # escalation-assessed.
        return Verdict(
            grounded=False, action=None, facts=facts, category=cat,
            escalate=False,
        )

    # -- planners -----------------------------------------------------------
    # Below this, the resolver did not recognise a name — it found the closest
    # thing to one. Anything reached by fuzzy matching alone scores here.
    _SURE_ENOUGH_TO_ACT = 0.88

    def _plan_launch(self, intent: Intent, facts: dict) -> Verdict:
        target = intent.slots.get("target", "")
        resolved = self.launcher.resolve(
            target, intent.slots.get("want_window", False), intent.slots.get("browser")
        )
        if resolved is None:
            return Verdict(True, None, facts, speak_bank="veto.not_found",
                           category="system")

        cap = "launch_urls" if resolved.kind == "url" else "launch_apps"
        perm = self.permission_for(cap)

        # Name it back and wait, when the match was a guess.
        #
        # Launching is configured to run without confirmation, and that is right
        # for the case it was configured for — "open Spotify" resolves through an
        # explicit alias and there is nothing to check. It is wrong for a target
        # that only came out of a fuzzy scan, because a spoken request cannot be
        # proof-read before it executes: by the time Jay hears which application
        # HIVE picked, it is already on screen. So a guess is spoken as a guess.
        if perm is Permission.ALLOW and resolved.score < self._SURE_ENOUGH_TO_ACT:
            from .phrasing import proper

            label = proper(resolved.label)
            return Verdict(True, ActionPlan(
                kind="launch", label=resolved.label,
                run=lambda r=resolved: self.launcher.launch(r),
                permission=Permission.CONFIRM, reversible=True,
                detail=" ".join(resolved.argv)), facts,
                speak_text=(f"Closest I've got to that is {label}. "
                            f"Want it open?"),
                category="system")
        plan = ActionPlan(
            kind="launch",
            label=resolved.label,
            run=lambda r=resolved: self.launcher.launch(r),
            permission=perm,
            reversible=True,
            detail=" ".join(resolved.argv),
        )
        facts["resolved_target"] = f"{resolved.label} ({resolved.kind})"
        # Name it. "Opening it now" was true of every launch HIVE has ever done,
        # which meant it confirmed nothing — hearing the app's name back is the
        # difference between an acknowledgement and a noise. Costs one synthesis
        # per app, then it is cached under that name for good.
        from .phrasing import acknowledgement

        said = acknowledgement(self.cfg, "ack.opening", target=resolved.label)
        return Verdict(True, plan, facts, speak_text=said or None,
                       speak_bank="ack.opening", category="system")

    def _plan_note(self, intent: Intent, facts: dict, stm) -> Verdict:
        content = (intent.slots.get("content") or "").strip()
        if not content:
            # "add that to my todo" with no payload — resolve from context rather
            # than writing an empty bullet.
            last = stm.last_user_text()
            return Verdict(True, None, facts,
                           speak_text="Add what, exactly?", category="writing")

        target = self.ltm.root / "20_Inbox"
        perm = self.vault_write_permitted(target)
        plan = ActionPlan(
            kind="note",
            label=f"note: {content[:60]}",
            run=lambda c=content: self._write_note(c),
            permission=perm,
            reversible=True,
        )
        return Verdict(True, plan, facts, speak_bank="ack.saved", category="writing")

    def _write_note(self, content: str) -> tuple[bool, str]:
        try:
            note = self.ltm.add_inbox(content)
            return True, note.path.name
        except Exception as exc:
            return False, str(exc)

    # -- requested write-ups -------------------------------------------------
    # The sections the model is asked to write, in the order they appear in the
    # file. Each one is a separate generation rather than one long prompt with
    # headings in it, for two reasons that both showed up in the vault: a small
    # model handed a multi-part instruction answers the first part and drifts,
    # and a single generation that fails takes the whole document with it. Five
    # requests that can fail independently produce a file that is thinner when
    # something goes wrong, instead of empty.
    # (heading, instruction, token budget, heavy model). The budgets are set by
    # what each section is worth rather than uniformly: the two short ones on the
    # ends are bookkeeping and run on the fast tier, and the three in the middle
    # are the document. Uniform budgets across five heavy generations put a
    # single request past three minutes of silent work, which is longer than any
    # announcement can carry.
    _DOC_SECTIONS = [
        ("In short",
         "Three or four sentences: what this conversation was actually about "
         "and what came out of it. No preamble.", 320, False),
        ("What was covered",
         "The substance, in full. Several paragraphs. Go through every subject "
         "that came up and say what was established about it — the claims, the "
         "reasoning, the specifics, the numbers and names. Do not compress. "
         "This section is the point of the document and it should be the "
         "longest thing in it.", 1600, True),
        ("Detail worth keeping",
         "The specifics a summary would normally drop: exact terms, "
         "definitions, distinctions, the reasons behind each conclusion, and "
         "anything technical that was said in passing. Write it as prose, one "
         "point per paragraph.", 1100, True),
        ("Raised but not chased",
         "Anything that came up and was left unfinished — questions asked and "
         "not answered, threads dropped, things mentioned once and moved past, "
         "and anything you know about these subjects that was relevant and did "
         "not get said out loud. Be explicit about which parts were not "
         "actually discussed.", 800, True),
        ("Where this could go next",
         "Two or three concrete next steps that follow from what was said. "
         "Actual actions, not encouragement.", 400, False),
    ]

    def _plan_document(self, intent: Intent, facts: dict, stm) -> Verdict:
        """Write a full markdown document into the vault, on request.

        This is the intent that did not exist. "Make a markdown file with
        everything we talked about" matched nothing in the grammar, so it fell
        through to conversation — and a conversational model asked to create a
        file says it is creating the file, which is the only thing it can do.
        Ask again and it says it again. Nothing was ever written, and there was
        no mechanism by which anything could have been.

        The property that replaces the promise: the completion sentence is
        *constructed from the file*. It is written, read back off disk, and its
        real name, line count and word count go into what HIVE says. There is no
        path through this code that reports a document that does not exist.

        It runs on its own thread because a full write-up is a couple of minutes
        of heavy-model generation — measured at 260 seconds for a five-section
        document — and holding the conversation for that is worse than useless
        when the request was "write that up" rather than "read that to me".
        """
        if stm.user_turns == 0:
            return Verdict(True, None, facts, category="writing", speak_text=(
                "There's nothing to write up yet — we haven't talked about "
                "anything this session."))

        subject = (intent.slots.get("subject") or "").strip()

        if self._writing is not None and self._writing.is_alive():
            return Verdict(True, None, facts, category="writing", speak_text=(
                "I'm still writing the last one. Give me a moment and ask "
                "again."))

        # A snapshot, taken now, on this thread. Two reasons and both matter.
        # The write runs for minutes while the conversation carries on
        # underneath it, so a live reference would document a conversation that
        # had not finished — and short-term memory is a plain dict that the turn
        # loop keeps writing to, so reading it from the writer thread is a race
        # that eventually raises mid-document.
        turns = list(stm.transcript())
        salient = stm.salient(24)

        def run() -> tuple[bool, str]:
            if self.llm is None or not self.llm.available():
                return False, ("I can't write that up — the language model "
                               "isn't reachable, and a write-up without it "
                               "would just be the transcript.")
            self._writing = threading.Thread(
                target=self._guarded_write, args=(subject, turns, salient),
                name="hive-writeup", daemon=True)
            self._writing.start()
            return True, (
                "On it. That's a couple of minutes of proper writing, so I'll "
                "carry on with you meanwhile and tell you what it's called when "
                "it lands.")

        plan = ActionPlan(
            kind="document",
            label=f"a write-up on {subject or 'this session'}",
            run=run,
            permission=self.vault_write_permitted(self.ltm.root / WRITEUPS),
            reversible=True,
        )
        return Verdict(True, plan, facts, category="writing")

    def _guarded_write(self, subject: str, turns: list,
                       salient: list[tuple[str, float]]) -> None:
        """Run the write and make sure a failure is audible rather than silent.

        A daemon thread that raises prints a traceback to a stderr nobody is
        watching — HIVE launches from the app grid with no terminal — and the
        symptom is indistinguishable from the bug this whole path was built to
        remove: a file that was promised and never appeared. So the one thing
        that must not happen here is failing quietly.
        """
        try:
            self._write_document(subject, turns, salient)
        except Exception as exc:
            import traceback

            self.bus.log(f"write-up crashed: {type(exc).__name__}: {exc}", "error")
            self.bus.log(traceback.format_exc()[-600:], "error")
            self._announce(False, "That write-up failed partway through. "
                                  "The error's in my terminal.")

    def wait_for_writes(self, timeout: float = 300.0) -> None:
        """Let an in-flight write-up finish before the process goes away.

        Called on shutdown. Without it, saying "write all that up" and then
        closing the window loses the file — which is the same broken promise
        this whole path exists to eliminate, arriving by a different route.

        The timeout is generous because the failure mode is total: the document
        is composed in memory and written once at the end, so a join that gives
        up early does not produce a shorter file, it produces no file. An
        earlier ninety seconds was measured as too short — a clean run takes
        about seventy, and a shutdown competing with consolidation for the same
        GPU takes considerably longer than that.
        """
        thread = self._writing
        if thread is None or not thread.is_alive():
            return
        self.bus.log("waiting for the write-up to finish", "memory")
        thread.join(timeout)
        if thread.is_alive():
            # Say so. Losing it quietly is the exact shape of the bug this
            # whole path was built to remove.
            self.bus.log(
                f"write-up still running after {timeout:.0f}s — it will be lost "
                f"when the process exits", "error")

    def _document_title(self, summary: str,
                        salient: list[tuple[str, float]]) -> str:
        """Name the write-up after what it turned out to be about.

        Asked of the fast model *after* the summary exists, because that is the
        first moment anything in the system knows. Salience alone produced
        "Prosody Emotional Contextual" and "Open Ejpt Focus" — top terms
        concatenated, which is a filename Jay would never guess and never find
        again. The salience version stays as the fallback, since a clumsy name
        still beats no file.
        """
        terms = [t for t, score in salient[:6] if score > 1.0 and len(t) > 3]
        fallback = (" ".join(w.title() for w in terms[:3]) if terms
                    else f"Session Notes {datetime.now():%d %B}")
        if not summary or self.llm is None:
            return fallback
        try:
            # The heavy tier, and it is free here: it has just written five
            # sections and is still resident. The fast tier returns
            # "ExamScheduling" — a real answer with the spaces missing, which is
            # the same failure the vault's topic resolver was built around.
            raw = self.llm.complete([
                {"role": "system", "content":
                    "You name documents. Reply with the title and nothing else: "
                    "two to five words, Title Case, real spaces between the "
                    "words, no punctuation, no quotes, no explanation. It "
                    "becomes a filename in an Obsidian vault, so it must read "
                    "like something a person filed."},
                {"role": "user", "content":
                    f"Name the document that opens with this:\n\n{summary[:700]}"},
            ], heavy=True, num_predict=32)
        except Exception:
            return fallback
        name = re.sub(r"[^\w\s&()-]", "", (raw or "").splitlines()[0] if raw else "")
        # "AIVoiceNotes" -> "AI Voice Notes". Cheap, and it recovers a good title
        # that would otherwise be thrown away for having one word in it.
        name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        words = name.split()
        return name if 1 < len(words) <= 6 and len(name) < 60 else fallback

    def _write_document(self, subject: str, turns: list,
                        salient: list[tuple[str, float]]) -> tuple[bool, str]:
        """Generate and write the document, then say where it landed.

        Runs on its own thread. The spoken result is deferred until the speaker
        is free, so the announcement lands in a gap in the conversation rather
        than on top of an answer.
        """
        transcript = "\n".join(
            f"{'Jay' if t.role == 'user' else 'HIVE'}: {t.text}"
            for t in turns if t.text
        )[-9000:]

        sections: list[tuple[str, str]] = []
        for heading, instruction, budget, heavy in self._DOC_SECTIONS:
            body = self._generate_section(heading, instruction, transcript,
                                          subject, budget, heavy)
            if body:
                sections.append((heading, body))
            self.bus.log(f"write-up: {heading} ({len(body)} chars)", "memory")

        recall = self._vault_recall(salient)
        if recall:
            sections.append(("What the vault already had on this", recall))

        summary = next((body for head, body in sections if head == "In short"), "")
        title = subject or self._document_title(summary, salient)

        try:
            note = self.ltm.write_document(
                title, sections, turns=turns, salient=salient,
                actions=list(self.actions_log),
                links=self.ltm.existing_topics()[:6], subject=subject,
            )
        except OSError as exc:
            self.bus.log(f"write-up failed: {exc}", "error")
            return self._announce(False, f"I couldn't write the file: {exc}")

        # Read it back. The spoken sentence is built from what is on disk, so a
        # write that silently produced nothing cannot be reported as a success.
        # This is the entire point of the path: HIVE used to say it had created
        # a file because saying so was all it did, and the only durable defence
        # against that is to make the sentence a function of the filesystem.
        try:
            written = note.path.read_text(encoding="utf-8")
        except OSError:
            return self._announce(False, "I wrote it and then couldn't read it back.")
        lines = written.count("\n") + 1
        words = len(written.split())
        if words < 120:
            return self._announce(False, (
                f"That one came out thin — only {words} words. It's at "
                f"{note.title} if you want it anyway."))

        self.bus.log(f"wrote {note.path.name} ({words} words)", "memory")
        return self._announce(True, (
            f"That write-up's done. It's in the vault under fifty Write-ups, "
            f"called {note.title} — {lines} lines, about "
            f"{round(words, -1):.0f} words. Summary at the top, then the detail, "
            f"then what we raised and didn't chase, and the whole transcript "
            f"underneath."))

    def _announce(self, ok: bool, line: str) -> tuple[bool, str]:
        """Say a result that arrived after its turn had already ended.

        Waits for a gap. The write finishes minutes later and the conversation
        has moved on, so speaking the moment it lands would cut across whatever
        HIVE is in the middle of saying. Bounded, because a gap that never comes
        means Jay is talking and the announcement has been overtaken anyway —
        the file is written either way, and the log records it.
        """
        if self.speaker is None:
            return ok, line
        for _ in range(240):                      # up to two minutes of waiting
            if self.speaker.closed:
                # The session ended while this was being written. The file is
                # on disk either way, which is the part that had to be true.
                self.bus.log(f"write-up landed after close: {line}", "memory")
                return ok, line
            if not self.speaker.speaking:
                break
            time.sleep(0.5)
        else:
            self.bus.log(f"write-up finished, never found a gap to say so: {line}",
                         "memory")
            return ok, line
        self.speaker.say(line, cached=False)
        return ok, line

    def _generate_section(self, heading: str, instruction: str,
                          transcript: str, subject: str,
                          budget: int = 1200, heavy: bool = True) -> str:
        """One section of the write-up. Returns "" rather than raising.

        Deliberately generous with tokens and deliberately not spoken. This runs
        after Jay has already been told it will take a minute, so the latency
        budget that governs every other path in the system does not apply, and
        the only thing left to optimise for is how much of the conversation
        survives into the file.
        """
        if self.llm is None or not self.llm.available():
            return ""
        focus = (f"Focus the whole document on: {subject}. " if subject else "")
        messages = [
            {"role": "system", "content":
                "You write documents for an Obsidian vault. Markdown prose for "
                "reading on a screen — not speech, so headings, emphasis and "
                "lists are all fine. Be specific and concrete: name the actual "
                "things discussed, quote the actual numbers. Never write "
                "'the conversation covered various topics'. Never pad. Never "
                "add a preamble or a sign-off. Write only the section asked "
                "for, with no heading of its own."},
            {"role": "user", "content":
                f"Here is the transcript of a voice session.\n\n{transcript}\n\n"
                f"{focus}Write the section titled '{heading}'.\n{instruction}"},
        ]
        try:
            body = self.llm.complete(messages, heavy=heavy,
                                     num_predict=budget).strip()
        except Exception as exc:
            self.bus.log(f"write-up section '{heading}' failed: {exc}", "warn")
            return ""
        # It writes the heading anyway, as a plain line or as its own markdown
        # header, and then the file has it twice. Strip it rather than prompt
        # harder — small models keep doing this and the same fix is already in
        # the session summariser for the same reason.
        # The comma matters: it writes "In short, the conversation discussed…",
        # and stripping only the heading leaves the section opening on a comma.
        body = re.sub(rf"(?i)^\s*#{{0,6}}\s*{re.escape(heading)}\s*[:.,;\-—]?\s*\n*",
                      "", body)
        return body.strip()

    def _vault_recall(self, salient: list[tuple[str, float]]) -> str:
        """What the vault already knew about the things just discussed.

        Deterministic — these are indexed mentions with dates, not the model's
        recollection — which is what makes it safe to put in a file that Jay
        will later read as a record.
        """
        if self.associations is None or not self.associations.ready:
            return ""
        from ..memory.associations import human_when

        seen: set[str] = set()
        out: list[str] = []
        for term, _score in salient[:8]:
            try:
                found = self.associations.connect(term, limit=3, strict=False)
            except Exception:
                continue
            for c in found:
                if c.display in seen:
                    continue
                seen.add(c.display)
                bit = (f"- **{c.display}** — {c.times} mention"
                       f"{'s' if c.times != 1 else ''} in the vault, last "
                       f"{human_when(c.days_ago)}, in `{c.source}`.")
                if c.open_loop:
                    bit += (f" Still open: you {c.loop_verb} it "
                            f"{human_when(c.loop_days)} with nothing recorded since.")
                if c.excerpt:
                    bit += f"\n  > {c.excerpt[:200].strip()}"
                out.append(bit)
        if not out:
            return ""
        return ("Indexed mentions from Jay's own notes, not recollection.\n\n"
                + "\n".join(out[:10]))

    def _plan_recall(self, intent: Intent, facts: dict) -> Verdict:
        query = intent.slots.get("query", "")
        hits = self.ltm.search(query, limit=4)
        if not hits:
            return Verdict(True, None, facts,
                           speak_text=f"Nothing in the vault about {query}.",
                           category="memory")
        # Hand the excerpts to the model as grounded context and let it phrase the
        # answer. The retrieval itself stays deterministic.
        facts["vault_hits"] = " || ".join(
            f"{p.stem}: {ex[:180]}" for p, ex in hits
        )
        return Verdict(grounded=False, action=None, facts=facts, category="memory")

    def _plan_system(self, intent: Intent, facts: dict) -> Verdict:
        perm = self.permission_for("system_query")
        question = intent.cleaned

        def run(q=question) -> tuple[bool, str]:
            reading = sysinfo.answer(q)
            if reading is None:
                snap = sysinfo.snapshot()
                reading = snap.get("disk")
            return (True, reading.spoken) if reading else (False, "no reading")

        plan = ActionPlan(
            kind="system", label="a system reading", run=run,
            permission=perm, reversible=True,
        )
        return Verdict(True, plan, facts, category="system")

    def _plan_status(self, facts: dict) -> Verdict:
        """The status briefing. Spoken from configuration, never from a guess.

        Deterministic — `speak_text`, so the model is not consulted at all. That
        is not only about the 300ms. The briefing makes specific claims about
        failures and bugs, and those claims are Jay's own authored values; handing
        them to a model to phrase invites it to round two into "a couple", or to
        explain a failure it knows nothing about. There is nothing here for
        language generation to add.

        Read `status.enabled = false` as "answer this like any other question"
        rather than as a refusal: the turn falls through ungrounded and the
        conversational path takes it, which is what happened before this existed.
        """
        if not self.cfg.get("status.enabled", True):
            return Verdict(grounded=False, action=None, facts=facts,
                           category="system")

        briefing = status.build(self.cfg)

        # The paths the spoken half deliberately leaves out. Same route the web
        # layer uses for the links HIVE does not read aloud — see
        # _publish_results, which exists for exactly this reason.
        if briefing.paths:
            self.bus.publish(TERMINAL_OUTPUT, text=briefing.terminal_text(),
                             kind="sys")

        # Ground the model for the turns that follow. Jay's next sentence is
        # usually about what was just read out, and without this the model is
        # answering questions about a briefing it never saw.
        facts["status_briefing"] = briefing.spoken

        return Verdict(True, None, facts, speak_text=briefing.spoken,
                       category="system")

    # Sites that accept a prefilled prompt in the query string. Anywhere else,
    # the question still goes to the clipboard so it is one paste away.
    _PREFILL = {
        "claude": "https://claude.ai/new?q={q}",
        "chatgpt": "https://chatgpt.com/?q={q}",
        "google": "https://www.google.com/search?q={q}",
        "perplexity": "https://www.perplexity.ai/search?q={q}",
    }

    def _plan_launch_ask(self, intent: Intent, facts: dict) -> Verdict:
        """Open something and put a question into it. Two effects, one action.

        The prefill parameter is best-effort — sites change their query handling
        without notice. The clipboard copy is not: it always works, so the
        question is never lost even when the URL parameter is ignored. HIVE says
        which of the two it is relying on rather than claiming success blindly.
        """
        from urllib.parse import quote_plus

        target = intent.slots.get("target", "")
        question = intent.slots.get("question", "")
        resolved = self.launcher.resolve(target, True, intent.slots.get("browser"))
        if resolved is None:
            return Verdict(True, None, facts, speak_bank="veto.not_found",
                           category="system")

        from ..skills.launcher import Target

        key = next((k for k in self._PREFILL if k in resolved.label.lower()
                    or k in target.lower()), None)
        prefilled = bool(key)
        if key:
            url = self._PREFILL[key].format(q=quote_plus(question))
            argv = self.launcher._browser_argv(url, intent.slots.get("browser"), True)
            resolved = Target("url", resolved.label, argv, resolved.score)

        def run(r=resolved, q=question) -> tuple[bool, str]:
            self.bus.publish(CLIPBOARD_SET, text=q)
            ok, label = self.launcher.launch(r)
            return ok, label

        plan = ActionPlan(kind="launch_ask", label=f"{resolved.label} with your question",
                          run=run, permission=self.permission_for("launch_urls"))
        short = question if len(question) < 70 else question[:67] + "..."
        said = (f"Opening {resolved.label} with the question already in it: {short}"
                if prefilled else
                f"Opening {resolved.label}. The question's on your clipboard — "
                f"paste it in.")
        return Verdict(True, plan, facts, speak_text=said, category="system")

    def _plan_jobs(self, intent: Intent, facts: dict) -> Verdict:
        """Open pre-filtered job searches, and read the resume if asked.

        Cannot read the listings themselves — no network, by design — so it does
        the part that does not need one and says plainly where the boundary is.
        """
        from ..skills import jobs

        role = intent.slots.get("role") or "help desk"
        location = intent.slots.get("location", "")
        radius = int(intent.slots.get("radius", 25))
        days = int(intent.slots.get("days", 7))
        sites = intent.slots.get("sites") or ["indeed", "ziprecruiter"]
        targets = jobs.build_urls(role, location, radius, days, sites)

        wants_resume = bool(re.search(
            r"\bresume|\bcv\b|qualif|do i meet|am i qualified|match", intent.cleaned, re.I))
        resume = jobs.find_resume() if wants_resume else None

        def run() -> tuple[bool, str]:
            from ..skills.launcher import Target
            opened = 0
            for i, t in enumerate(targets):
                # First search gets a new window; the rest become tabs in it.
                argv = self.launcher._browser_argv(t.url, None, i == 0)
                ok, _ = self.launcher.launch(Target("url", t.label, argv, 1.0))
                opened += int(ok)
            return (opened > 0), f"{opened} search(es)"

        plan = ActionPlan(kind="jobs", label="the job searches", run=run,
                          permission=self.permission_for("launch_urls"))

        said = jobs.spoken_summary(role, location, radius, days, targets)
        if wants_resume:
            if resume is None:
                said += (" I couldn't find a resume PDF in Documents, Downloads or "
                         "Desktop — grant me the folder in my terminal and I'll read it.")
            else:
                text = jobs.read_pdf(resume)
                prof = jobs.profile_resume(text)
                facts["resume_path"] = str(resume)
                facts["resume_signals"] = ", ".join(prof["found"]) or "none detected"
                facts["resume_text"] = text[:4000]
                said += (f" I've read {resume.name}. It shows "
                         f"{', '.join(prof['found'][:5]) or 'little I can match'}.")
            # The honest boundary, said once and not repeated every turn.
            said += (" I can't read the listings themselves — that needs network "
                     "access I don't have. Paste one in and I'll tell you how you "
                     "line up against it.")
        return Verdict(True, plan, facts, speak_text=said, category="research")

    # -- the network ---------------------------------------------------------
    def _publish_results(self, query: str, results) -> None:
        """Hand structured results to whoever is collecting them.

        Published rather than returned because two very different things want
        them and neither is in the call chain: the terminal pane, which prints
        the links HIVE deliberately does not read aloud, and HIVE-Q's runner,
        which folds them into the task's markdown log. An event is how the rest
        of this system already solves that.
        """
        if not results:
            return
        payload = [{"title": r.title, "url": r.url, "snippet": r.snippet,
                    "kind": r.kind, "meta": r.meta} for r in results]
        self.bus.publish(WEB_RESULTS, query=query, results=payload)
        lines = [f"  {r['title'][:72]}" + (f"\n    {r['url']}" if r["url"] else "")
                 for r in payload[:8]]
        self.bus.publish(TERMINAL_OUTPUT,
                         text=f"results for {query[:60]}:\n" + "\n".join(lines),
                         kind="plan")

    def _offline_fallback(self, facts: dict, category: str) -> Verdict:
        """No network. Behave exactly as every version before this one did.

        Returned ungrounded, so the turn carries on to the generative path and
        HIVE answers from what it knows — which is the right answer to "what's
        worth reading on X" even with a network, and was the only answer
        available before. `base_facts` has already told the model it is offline,
        so it will say so rather than promising a search.
        """
        return Verdict(grounded=False, action=None, facts=facts, category=category)

    def _plan_lookup(self, intent: Intent, facts: dict) -> Verdict:
        """Go and read: papers for a research request, the open web otherwise."""
        from ..skills import web as webskill

        is_research = intent.name == "research"
        topic = (intent.slots.get("topic") or intent.slots.get("query") or "").strip()
        if not topic or self.web is None or not self.web.online():
            return self._offline_fallback(facts, "research")

        def run(q=topic, research=is_research) -> tuple[bool, str]:
            try:
                if research:
                    spoken, results = webskill.research(
                        self.web, q, limit=int(self.cfg.get("net.max_results", 6)) - 2)
                else:
                    spoken, results = webskill.look_up(self.web, self.llm, q)
            except Exception as exc:
                return False, (f"The search fell over — {type(exc).__name__}. "
                               f"I'd rather say that than make something up.")
            self._publish_results(q, results)
            return True, spoken

        plan = ActionPlan(
            kind="search",
            label=f"{'papers' if is_research else 'a search'} on {topic}",
            run=run, permission=self.permission_for("network_request"),
            reversible=True,
        )
        facts["search_topic"] = topic
        return Verdict(True, plan, facts, category="research")

    def _plan_site_search(self, intent: Intent, facts: dict) -> Verdict:
        """Open a site's own search, already filled in — and read it back when
        the request was for an answer rather than for a page.

        Two effects on purpose, and the split matters. Opening the pre-filled URL
        is the literal request ("type this into the search bar") and it always
        works, because it is a URL. Reading the results is the useful half of a
        shopping or reference question, and it comes from a `site:` search rather
        than from the retailer's own page — see skills/web.py for why fetching
        those directly produces a confident summary of a bot check.
        """
        from ..skills import web as webskill

        key = intent.slots.get("site", "")
        query = intent.slots.get("query", "")
        spec = (self.cfg.apps.get("search_sites") or {}).get(key)
        if not spec or not query:
            return Verdict(True, None, facts, speak_bank="veto.not_found",
                           category="system")

        label = (spec.get("aliases") or [key])[0]
        url = webskill.site_search_url(spec.get("url", ""), query)
        # Reading is requested by phrasing, but a site marked read:false is never
        # read regardless — six video titles spoken aloud is not an answer.
        read = bool(intent.slots.get("read")) and bool(spec.get("read"))
        online = self.web is not None and self.web.online()

        def run() -> tuple[bool, str]:
            from ..skills.launcher import Target
            argv = self.launcher._browser_argv(url, None, True)
            ok, _ = self.launcher.launch(Target("url", label, argv, 1.0))
            opened = (f"{label} is up with {query} in the search."
                      if ok else f"I couldn't open {label}.")
            if not (read and online):
                return ok, opened
            from ..net import Offline
            try:
                spoken, results = webskill.product_options(
                    self.web, query, site_label=label,
                    site_domain=spec.get("domain", ""))
            except Offline as exc:
                # The page opened, so half the request landed — but the half
                # that needed reading did not, and reporting the whole task as
                # done would bury exactly the thing worth retrying. Fails, with
                # the page still open and the reason said out loud.
                return False, (f"{opened} I couldn't read the listings though — "
                               f"{exc}. Worth another go in a few minutes.")
            except Exception as exc:
                return False, (f"{opened} I couldn't read the listings — "
                               f"{type(exc).__name__}.")
            self._publish_results(f"{label}: {query}", results)
            return ok, spoken

        plan = ActionPlan(kind="site_search", label=f"{label} for {query}",
                          run=run, permission=self.permission_for("launch_urls"))
        facts["site_search"] = f"{label}: {query}"
        # What is being looked for, phrased so it can be spoken as-is. The
        # callosum reads this to build the holding line for a slow action, and
        # "best buy: keyboards" is a log entry, not a sentence.
        facts["search_topic"] = query
        # Only the non-reading path gets a canned line; the reading path's
        # sentence is built from what actually came back.
        said = None if (read and online) else f"Searching {label} for {query}."
        return Verdict(True, plan, facts, speak_text=said, category="research")

    # -- HIVE-Q --------------------------------------------------------------
    def _plan_query(self, intent: Intent, facts: dict) -> Verdict:
        """Voice control for the queue. The override lives here as well as in
        the terminal, because "end query" is the one thing you may need to say
        while HIVE is in the middle of saying something else."""
        if self.query is None:
            return Verdict(True, None, facts, category="system", speak_text=(
                "HIVE-Q isn't running in this session."))

        if intent.name == "query_end":
            if not self.query.running:
                return Verdict(True, None, facts, category="system",
                               speak_text="There's no query running.")

            def stop() -> tuple[bool, str]:
                return True, self.query.end("you said so")

            return Verdict(True, ActionPlan(
                kind="query", label="ending the query", run=stop,
                permission=Permission.ALLOW), facts, category="system")

        if intent.name == "query_run":
            def go() -> tuple[bool, str]:
                ok, msg = self.query.start()
                return ok, msg

            return Verdict(True, ActionPlan(
                kind="query", label="running the query", run=go,
                permission=Permission.ALLOW), facts, category="system")

        def show() -> tuple[bool, str]:
            self.bus.publish(QUERY_WINDOW, open=True)
            return True, "HIVE-Q is up. Type your list and hit run."

        return Verdict(True, ActionPlan(
            kind="query", label="the HIVE-Q window", run=show,
            permission=Permission.ALLOW), facts, category="system")

    def _plan_workspace(self, intent: Intent, facts: dict) -> Verdict:
        """Act on folders granted from the terminal.

        Folder names are matched against the grant list rather than parsed out of
        the transcript. "hyperintelligent_agent" is not something Whisper renders
        reliably, so matching known names against the utterance succeeds where
        extracting an arbitrary one would produce a plausible wrong directory.
        """
        ws = self.workspace
        if ws is None or not ws.grants:
            # The offer follows "What I can do is", so it has to be something
            # HIVE does, not an instruction to Jay — and it is backed by a real
            # action, so saying yes actually opens the pane.
            def open_term() -> tuple[bool, str]:
                self.bus.publish(TERMINAL_TOGGLE, open=True)
                return True, "terminal open"

            return Verdict(True, None, facts, category="analysis", refusal=Refusal(
                reason="no folders have been granted to me yet",
                offer=("open my terminal for you — then type grant, and the folder "
                       "name. Grants are typed rather than spoken, so a misheard "
                       "name can't point me at the wrong directory. Want it open?"),
                action=ActionPlan(kind="terminal", label="the HIVE terminal",
                                  run=open_term, permission=Permission.ALLOW),
            ))

        if intent.name == "workspace_list":
            names = ", ".join(g.path.name for g in ws.grants.values())
            return Verdict(True, None, facts, category="analysis",
                           speak_text=f"I have access to {names}.")

        said = intent.cleaned
        mentioned = [g for key, g in ws.grants.items()
                     if key in said or g.path.name.lower().replace("_", " ") in said]

        if intent.name == "workspace_describe":
            if not mentioned:
                return self._workspace_ambiguous(ws, facts)
            s = ws.structure(mentioned[0].path.name)
            if not s:
                return Verdict(True, None, facts, speak_bank="veto.not_found",
                               category="analysis")
            facts["workspace_detail"] = str(s)
            top = ", ".join(s["dirs"][:6]) or "no subfolders"
            return Verdict(True, None, facts, category="analysis", speak_text=(
                f"{s['name']} has {s['file_count']} files across "
                f"{s['dir_count']} folders. Top level: {top}."))

        # organize A like B
        if len(mentioned) < 2:
            return self._workspace_ambiguous(ws, facts)
        source, template = mentioned[0], mentioned[1]
        ok, preview, plan = ws.plan_mirror(source.path.name, template.path.name)
        if not ok:
            return Verdict(True, None, facts, category="analysis", refusal=Refusal(
                reason=preview,
                offer=("grant it for writing in my terminal with "
                       f"`grant --write {source.path.name}`, then ask me again."),
            ))

        self.bus.publish(TERMINAL_OUTPUT, text=preview, kind="plan")

        def run(p=plan) -> tuple[bool, str]:
            return ws.apply_mirror(p)

        action = ActionPlan(
            kind="workspace", label=f"reorganising {source.path.name}",
            run=run, permission=Permission.CONFIRM, reversible=False,
        )
        n_dirs = len(plan[0][1]) if plan else 0
        n_moves = len(plan[0][2]) if plan else 0
        return Verdict(True, action, facts, category="analysis", speak_text=(
            f"I've put the plan in the terminal: {n_dirs} folders to create and "
            f"{n_moves} files to move in {source.path.name}. This isn't reversible. "
            f"Say go and I'll run it."))

    def _workspace_ambiguous(self, ws, facts: dict) -> Verdict:
        names = ", ".join(g.path.name for g in ws.grants.values())
        return Verdict(True, None, facts, category="analysis", refusal=Refusal(
            reason="I couldn't tell which folders you meant",
            offer=f"name two of these: {names}.",
        ))

    def _plan_media(self, intent: Intent, facts: dict) -> Verdict:
        """Media transport, and an honest answer when transport is not enough."""
        from ..skills import media

        name = intent.name
        running = media.players()
        if not running:
            return Verdict(True, None, facts, category="system", refusal=Refusal(
                reason="nothing's playing and no player is open",
                offer="open Spotify for you. Say the word.",
            ))

        if name == "media_now":
            np_ = media.now_playing()
            return Verdict(True, None, facts, category="system",
                           speak_text=np_.spoken if np_ else "Nothing's playing.")

        # A named track or playlist. HIVE can drive the transport but cannot
        # search a library — that needs the Spotify Web API, an account token and
        # network access, none of which it has. Say so, and offer the search.
        if name == "media_named":
            what = intent.slots.get("what", "")
            uri = media.search_uri(what)

            def run_search(u=uri) -> tuple[bool, str]:
                ok, _ = media.open_uri(u)
                return (ok, what) if ok else (False, "Spotify refused the search")

            plan = ActionPlan(
                kind="media_search", label=f"a Spotify search for {what}",
                run=run_search, permission=Permission.ALLOW,
            )
            # Phrased as a continuation of the lead ("I can't do that one — ..."),
            # not as its own sentence. A reason that starts with "I can't" makes
            # the rendered line stutter: "I can't do that one, I can't search...".
            return Verdict(True, None, facts, category="system", refusal=Refusal(
                reason=(f"searching your library needs Spotify's API and an "
                        f"account, and I run entirely offline"),
                offer=f"put a Spotify search for {what} in front of you. Want that?",
                action=plan,
            ))

        fn = {
            "media_play": media.play,
            "media_pause": media.pause,
            "media_next": lambda: media.skip(backwards=False),
            "media_prev": lambda: media.skip(backwards=True),
        }.get(name)
        if fn is None:
            return Verdict(True, None, facts, speak_bank="veto.not_found",
                           category="system")

        plan = ActionPlan(kind="media", label="playback", run=fn,
                          permission=Permission.ALLOW)
        return Verdict(True, plan, facts, category="system")

    def _plan_voice(self, intent: Intent, facts: dict) -> Verdict:
        if self.speaker is None:
            return Verdict(True, None, facts, speak_bank="veto.error", category="system")
        req = intent.slots.get("request", "").lower()
        voices = self.speaker.available_voices()

        chosen = next((v for v in voices if v.lower() in req), None)
        if chosen is None:
            if "american" in req:
                chosen = "af_heart" if "af_heart" in voices else None
            elif "british" in req or "english" in req:
                chosen = "bf_emma" if "bf_emma" in voices else None
            elif any(w in req for w in ("lower", "softer", "quieter", "calm")):
                chosen = "af_nicole" if "af_nicole" in voices else None
        if chosen is None:
            return Verdict(True, None, facts,
                           speak_text="Which voice? Say a name, or American or British.",
                           category="system")

        def run(v=chosen) -> tuple[bool, str]:
            return (True, v) if self.speaker.set_voice(v) else (False, v)

        plan = ActionPlan(kind="voice", label=f"the {chosen} voice", run=run,
                          permission=Permission.ALLOW, reversible=True)
        return Verdict(True, plan, facts,
                       speak_text=f"Switching to {chosen.replace('_', ' ')}.",
                       category="system")

    # -- claim verification (the last line) ---------------------------------
    _DATE_CLAIM = re.compile(
        r"\b(?:today is|it'?s|it is|the date is)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|"
        r"november|december|\d{4})", re.I)

    # Things HIVE cannot do under any circumstances. Reaching the network
    # changed what it can read; it changed nothing about what it can *send*,
    # and nothing about Wayland.
    _CAPABILITY_CLAIM = re.compile(
        r"\bI (?:have|can|will|just|already)\s+(?:now\s+)?"
        r"(sent an email|emailed|called|texted|posted|tweeted|messaged|"
        r"logged in|signed in|placed an order|ordered|bought|purchased|"
        r"filled (?:in|out) the form|submitted|installed|"
        r"moved the window|focused|resized|minimi[sz]ed|closed your)", re.I)

    # Things that are true only when there is a network. Left in the verifier
    # rather than deleted: "I googled that" is still a lie when the machine is
    # offline, and offline is a state HIVE spends real time in. What changed is
    # that the answer now depends on a fact rather than being a constant.
    _NETWORK_CLAIM = re.compile(
        r"\bI (?:have|can|will|just|already)\s+(?:now\s+)?"
        r"(searched the web|googled|browsed|looked online|looked it up online|"
        r"downloaded|read the (?:page|article|paper)|fetched)", re.I)

    def verify(self, text: str, facts: dict) -> str | None:
        """Re-check assertions against ground truth. Returns a veto reason or None.

        Runs on generated text before it is spoken. Narrow by design: it only
        challenges claims we can actually adjudicate. A verifier that guesses is
        just a second hallucination.

        The network split below is the reason this method now takes `facts`
        seriously rather than only the clock. Before HIVE went online, "I
        searched the web" was false by construction and could be vetoed without
        looking anything up. It is now false only when the machine is offline —
        and a verifier that kept the old rule would veto HIVE for accurately
        describing what it had just done, which is a worse failure than the one
        the rule was written to catch.
        """
        now = datetime.now()

        m = self._DATE_CLAIM.search(text)
        if m:
            claimed = m.group(1).lower()
            truth = f"{now:%A %B %Y}".lower()
            if claimed not in truth and claimed != str(now.year):
                return f"asserted date '{claimed}' contradicts {now:%A, %B %d %Y}"

        m = self._CAPABILITY_CLAIM.search(text)
        if m:
            return f"claimed a capability it does not have: '{m.group(1)}'"

        m = self._NETWORK_CLAIM.search(text)
        if m and facts.get("network") != "online":
            return f"claimed to have been online: '{m.group(1)}'"

        return None
