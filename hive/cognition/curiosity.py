"""Curiosity — the one thing HIVE volunteers, and the rules that keep it rare.

Two behaviours live here because they compete for the same slot, and only one of
them can have it in any given turn:

  connection   something in this turn that the vault has seen before, especially
               something started and never closed out. "You applied there three
               weeks ago and nothing came back — worth another pass?"
  probe        a question about what was just discussed, aimed at the reason
               behind it rather than the content of it. "What drew you to
               Kerberoasting?" — the kind of question that starts a thread
               instead of closing one.

Connections outrank probes: one is new information, the other is conversation.

The hard part is not generating either. It is not doing it constantly. An
assistant that asks a follow-up after every answer is an interrogation, and the
first time it asks about something you did not care about you stop listening to
the ones you would have. So the gates are deliberately unforgiving:

  - never after an instruction. "Open Spotify" is not an invitation to talk.
  - never twice running, and never about a topic already asked about this session.
  - never when the operator has asked for brevity, and no probes late at night.
  - never on a single sighting: an entity earns a mention by recurring or by
    carrying an unfinished commitment, not by existing.
  - never a connection HIVE has to reach for. If the model already worked the
    recall into its own answer, the aside is dropped rather than repeated.

The model is asked for the question, and a deterministic line stands behind it.
Small models produce "Let me know if you have any other questions!" with great
enthusiasm; when that happens the line built here is spoken instead, because it
is seeded with entities that were actually named and cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..memory.associations import Connection, human_when

# Intents that are instructions, not conversation. Nothing is volunteered after
# one: the operator asked for a thing to happen and is waiting for it to happen.
_NEVER = {
    "presence", "presence_bare", "time", "stop", "sleep", "shutdown", "yes", "no",
    "launch", "launch_ask", "note_add", "note_write", "voice_change", "voice_list",
    "night_mode", "terminal_open", "terminal_close", "workspace_list", "misheard",
    "empty", "plan",
}

# Turns where a connection is worth raising even though no language model is
# involved — the answer is deterministic, and the recall is the valuable half.
_CONNECT_ANYWAY = {"job_search", "note_recall", "workspace_describe"}

_QUESTION_TAIL = re.compile(r"\?\s*[\"'’)]*\s*$")

# Words that recur in a transcript without ever being what the transcript is
# about. Two families: containers, which name the *form* of the thing rather
# than the thing ("articles", "notes", "version"), and comparatives, which are
# always modifiers ("smoother", "better", "faster"). Both score well on
# salience, and neither can finish the sentence "what drew you to ___?".
_NOT_A_SUBJECT = {
    "article", "articles", "paper", "papers", "source", "sources", "note",
    "notes", "file", "files", "thing", "things", "stuff", "part", "parts",
    "point", "points", "version", "versions", "example", "examples", "list",
    "lists", "question", "questions", "answer", "answers", "detail", "details",
    "summary", "topic", "topics", "session", "sessions", "conversation",
    "information", "content", "context", "reason", "reasons", "idea", "ideas",
    "option", "options", "case", "cases", "issue", "issues", "problem",
    "problems", "result", "results", "step", "steps", "sound", "sounds",
    "sounding", "human", "people", "person", "someone", "something",
}
# Comparatives and superlatives, which are always modifiers. Listed rather than
# matched on a suffix, because the suffix rule that catches "smoother" also
# catches "container", "parameter" and "kerberoasting" — and a gerund is very
# often exactly the subject worth asking about in this domain.
_ADJECTIVAL = {
    "smoother", "smoothest", "better", "best", "worse", "worst", "faster",
    "fastest", "slower", "easier", "easiest", "harder", "hardest", "longer",
    "shorter", "bigger", "biggest", "smaller", "cleaner", "closer", "further",
    "higher", "lower", "newer", "newest", "older", "oldest", "stronger",
    "deeper", "simpler", "quicker", "louder", "quieter", "nicer", "safer",
    "really", "actually", "basically", "probably", "definitely", "usually",
}


def _is_subject(term: str, said: str = "") -> bool:
    """Could this word plausibly finish "what drew you to ___?"

    Two tests. The word has to be a *thing* rather than a container or a
    modifier, and — when the current utterance is available — Jay has to have
    just said it. Session-wide salience alone is too weak a signal to build a
    question on: it ranks whatever recurred, so in "pull up articles on smoother
    voice agents" the top terms are "articles" and "smoother", and the question
    that came out was "Which of articles and smoother do you actually want to go
    deeper on?"

    Deliberately conservative in the rejecting direction. A missed probe costs
    one silence; a probe about "smoother" costs the credibility of every probe
    after it, because it proves the questions are generated rather than meant.
    """
    t = (term or "").strip().lower()
    if len(t) < 5 or t in _NOT_A_SUBJECT or t in _ADJECTIVAL:
        return False
    return not said or t in said.lower()


@dataclass
class Aside:
    """Something to say after the answer, and what to tell the model beforehand."""

    kind: str                       # "connection" | "probe"
    text: str = ""                  # the deterministic line, if the model declines
    directive: str = ""             # appended to the system prompt
    fact: str = ""                  # grounded recall, injected as a fact
    topic: str = ""
    other: str = ""
    connections: list[Connection] = field(default_factory=list)


def _fill(template: str, **slots: object) -> str:
    """Substitute {slots} without str.format's habit of raising on the ones a
    given line in the persona bank happens not to use."""
    out = template
    for key, value in slots.items():
        out = out.replace("{" + key + "}", str(value))
    return re.sub(r"\s+", " ", out).strip()


class Curiosity:
    def __init__(self, cfg, bus, associations, subjective, night=None):
        self.cfg, self.bus = cfg, bus
        self.assoc = associations
        self.subjective = subjective
        self.night = night
        self.enabled = bool(cfg.get("curiosity.enabled", True))
        self._gap = int(cfg.get("curiosity.min_turns_between", 2))
        self._min_words = int(cfg.get("curiosity.min_words", 6))

        self._turn = 0
        self._last_probe_turn = -99
        self._asked_topics: set[str] = set()
        self._raised: set[str] = set()      # entity keys already connected this session
        self._last_was_question = False

    # -- observation --------------------------------------------------------
    def note_reply(self, spoken: str) -> None:
        """Remember whether HIVE's last utterance ended on a question.

        If it did, the operator's next turn is an answer to it. Asking a second
        question on top of an answer to the first is how a conversation turns
        into a form.
        """
        self._last_was_question = bool(_QUESTION_TAIL.search(spoken or ""))

    # -- decision -----------------------------------------------------------
    def consider(self, intent, text: str, stm) -> Aside | None:
        """Decide what, if anything, to volunteer this turn."""
        if not self.enabled:
            return None
        self._turn += 1
        name = intent.name
        if name in _NEVER or name.startswith("media_"):
            return None

        connections = self._connections(text)
        if connections:
            return self._connection_aside(connections)

        if name not in _CONNECT_ANYWAY:
            probe = self._probe_aside(intent, text, stm)
            if probe is not None:
                return probe
        return None

    # -- connections --------------------------------------------------------
    def _connections(self, text: str) -> list[Connection]:
        try:
            found = self.assoc.connect(text, limit=3, exclude=self._raised)
        except Exception as exc:
            self.bus.log(f"association lookup failed: {exc}", "warn")
            return []
        # Earn the interruption: an unfinished commitment qualifies once it has
        # had time to come back — asking whether this morning's application has
        # been answered is not memory, it is nagging — and anything else has to
        # be a genuinely recurring thread.
        return [c for c in found
                if (c.open_loop and c.loop_days >= 2)
                or (c.times >= 3 and c.days_ago >= 3)]

    def _connection_aside(self, connections: list[Connection]) -> Aside:
        from ..memory.associations import normalise

        for c in connections:
            self._raised.add(normalise(c.display))

        loops = [c for c in connections if c.open_loop]
        bridged = [c for c in loops if c.via]
        if bridged:
            # Reached through the subject rather than named. The phrasing has to
            # say what the connection *is*, because he never mentioned these.
            names = self._and_join([c.display for c in bridged[:3]])
            line = _fill(self.cfg.line("connection.related"),
                         topic=names, other=bridged[0].via,
                         verb=bridged[0].loop_verb,
                         when=human_when(min(c.loop_days for c in bridged)),
                         n=len(bridged))
        elif len(loops) >= 2:
            names = self._and_join([c.display for c in loops[:3]])
            line = _fill(self.cfg.line("connection.overlap"),
                         n=len(loops), topic=names,
                         when=human_when(min(c.loop_days for c in loops)))
        elif loops:
            c = loops[0]
            bank = "connection.stale" if c.loop_days >= 10 else "connection.open_loop"
            line = _fill(self.cfg.line(bank), topic=c.display, verb=c.loop_verb,
                         when=human_when(c.loop_days), n=c.times)
        else:
            c = connections[0]
            line = _fill(self.cfg.line("connection.seen_before"), topic=c.display,
                         when=human_when(c.days_ago), n=self._ordinal(c.times))

        fact = self.assoc.fact_line(connections)
        return Aside(
            kind="connection",
            text=line,
            fact=fact,
            topic=connections[0].display,
            connections=connections,
            directive=(
                "RECALL — this comes from Jay's own vault and is verified. "
                f"{fact}. If it genuinely bears on what he just asked, work it "
                "into your answer in one clause. If it does not bear on it, "
                "ignore this entirely — never force a connection to fit. Either "
                "way, do not close with a question; one is added after you."
            ),
        )

    # -- probes -------------------------------------------------------------
    def _probe_aside(self, intent, text: str, stm) -> Aside | None:
        if self._last_was_question:
            return None
        # Terse by explicit request — the one signal that switches this off
        # rather than merely slowing it down.
        if self.subjective.model.verbosity < 0.5:
            return None
        # Late at night, and in night mode, questions get rarer but do not stop.
        # An earlier version blocked them outright, which would have disabled the
        # feature across most of the hours this machine is actually used.
        gap = self._gap
        if (self.cfg.timeslot() == "late_night"
                or (self.night is not None and getattr(self.night, "active", False))):
            gap *= 2
        if self._turn - self._last_probe_turn <= gap:
            return None
        # Substance test. "what time is it" and "thanks" are not openings.
        words = len(text.split())
        if words < self._min_words and "?" not in text:
            return None

        topic, other = self._topics(text, stm)
        if not topic:
            return None
        key = topic.lower()
        if key in self._asked_topics:
            return None
        self._asked_topics.add(key)
        self._last_probe_turn = self._turn

        bank = "curiosity.bridge" if other else "curiosity.probe"
        line = _fill(self.cfg.line(bank), topic=topic, other=other or topic)
        return Aside(
            kind="probe",
            text=line,
            topic=topic,
            other=other,
            # The model is told to stay out of it.
            #
            # It was originally asked to write the follow-up itself, on the theory
            # that it has the whole answer in front of it. In practice a local
            # model asked for a curious question produces "What specific area
            # would you like to explore next?" — grammatically a question, and
            # hollow. Worse, once it has been streamed it has already been
            # spoken, so there is no way to replace it with a better one. So the
            # question is built here, from something Jay actually named, and the
            # model's only instruction is not to get there first.
            directive=(
                "ENDING — answer and stop. Do not close with a question, an "
                "offer of further help, or 'let me know'. A follow-up question "
                "is added after you and two would be one too many."
            ),
        )

    def _topics(self, text: str, stm) -> tuple[str, str]:
        """The one or two things worth being curious about, from this turn.

        Entities the operator actually named beat whatever is merely salient:
        asking about a term he used is interested, asking about a term the
        session statistics surfaced is uncanny.
        """
        named: list[str] = []
        try:
            # Prefer the vault's spelling of a name over the transcript's.
            # Whisper writes "tryhackme"; the notes say "TryHackMe", and the
            # terminal mirrors these lines verbatim.
            named = [self.assoc.best_form(n) for n in self.assoc.entities_in(text)]
        except Exception:
            named = []
        # Longest first — a multiword name is more specific, and specificity is
        # the whole difference between a good question and a filler one.
        named = sorted({n for n in named if len(n) > 3},
                       key=lambda n: (-len(n.split()), -len(n)))
        picked = [n for n in named if n.lower() not in self._asked_topics]
        if not picked:
            # Salience is a fallback and it is a weak one: it ranks whatever
            # words recurred, and in a turn like "pull up articles on smoother
            # voice agents" the words that recur are "articles" and "smoother".
            # Asked back, that produced "Which of articles and smoother do you
            # actually want to go deeper on?" — a question about nothing, in a
            # session where HIVE otherwise did well. A salient term only earns
            # the slot if it is a plausible *subject*, which a bare adjective or
            # a container noun is not.
            picked = [t for t, score in stm.salient(6)
                      if score > 1.4 and _is_subject(t, text)]
        if not picked:
            return "", ""
        return picked[0], (picked[1] if len(picked) > 1 else "")

    # -- delivery -----------------------------------------------------------
    def resolve(self, aside: Aside | None, answer: str, bank: str = "") -> str:
        """What actually gets said after the answer. '' when nothing should be.

        Dropped when the model already did the job itself — either by ending on
        a question of its own despite being told not to, or, for a connection, by
        naming the thing HIVE was about to bring up. Saying it twice is worse
        than not saying it at all, and a small model ignoring an instruction is
        a thing to absorb rather than a thing to be surprised by.
        """
        if aside is None or not aside.text:
            return ""
        low = (answer or "").lower()
        if _QUESTION_TAIL.search(answer or ""):
            self.bus.log("aside dropped: the answer already ended on a question",
                         "memory")
            return ""
        if aside.kind == "connection":
            for c in aside.connections:
                if c.display.lower() in low:
                    return ""
            return aside.text

        # A calibrated turn asks a calibrated question. HackBot wonders whether
        # he has run the thing yet; VOSS wonders what the pattern underneath is.
        # The gating above is identical — only the phrasing moves.
        if bank and aside.topic and self.cfg.bank(bank):
            line = _fill(self.cfg.line(bank), topic=aside.topic,
                         other=aside.other or aside.topic)
            if line:
                return line
        return aside.text

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _and_join(names: list[str]) -> str:
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    @staticmethod
    def _ordinal(n: int) -> str:
        return {2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}th")
