"""Subjective hemisphere — personalisation, tone, and the user model.

This half decides *how* things are said and *what is worth volunteering*. It
never decides whether something is true or permitted; that authority sits
entirely with the objective hemisphere.

It maintains a persistent model of the operator across sessions, stored in the
vault at `HIVE/_state/preferences.json`. The signals it tracks are the ones that
can be observed without asking:

  verbosity      inferred from explicit corrections ("shorter", "more detail")
  habits         which applications get opened, and at what hours
  recurrence     which topics keep coming back
  rhythm         when sessions actually happen

The adaptation is deliberately slow and bounded. A preference model that swings
on a single utterance produces an assistant that feels erratic rather than
attentive — so every signal nudges, none of them jump, and everything is clamped.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from .intents import Intent

# Explicit steering. These are the moments the operator tells you directly how
# they want to be spoken to, and they should count far more than anything inferred.
_SHORTER = re.compile(
    r"\b(shorter|too long|too much|be brief|keep it (short|brief)|less detail|"
    r"get to the point|tl;?dr|summar(ise|ize) that|cut it down)\b", re.I)
_LONGER = re.compile(
    r"\b(more detail|elaborate|expand|go deeper|tell me more|full version|"
    r"explain (more|further|properly)|walk me through)\b", re.I)
_SLOWER = re.compile(r"\b(slow(er| down)|too fast)\b", re.I)
_FASTER = re.compile(r"\b(faster|speed up|too slow)\b", re.I)


@dataclass
class StyleFrame:
    tone: str = "crisp"
    verbosity: int = 1            # 0 terse · 1 normal · 2 full
    time_flavor: str = ""
    proactive: str | None = None
    sentence_budget: str = "two or three sentences"


@dataclass
class UserModel:
    verbosity: float = 1.0        # continuous, rounded only at use time
    speech_rate: float = 1.06
    app_counts: dict[str, int] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    hour_counts: dict[str, int] = field(default_factory=dict)
    sessions: int = 0
    total_turns: int = 0
    last_session: str = ""
    voice: str = ""

    def to_json(self) -> dict:
        return {
            "verbosity": round(self.verbosity, 3),
            "speech_rate": round(self.speech_rate, 3),
            "app_counts": self.app_counts,
            "topic_counts": self.topic_counts,
            "hour_counts": self.hour_counts,
            "sessions": self.sessions,
            "total_turns": self.total_turns,
            "last_session": self.last_session,
            "voice": self.voice,
        }

    @classmethod
    def from_json(cls, d: dict) -> "UserModel":
        m = cls()
        m.verbosity = float(d.get("verbosity", 1.0))
        m.speech_rate = float(d.get("speech_rate", 1.06))
        m.app_counts = dict(d.get("app_counts", {}))
        m.topic_counts = dict(d.get("topic_counts", {}))
        m.hour_counts = dict(d.get("hour_counts", {}))
        m.sessions = int(d.get("sessions", 0))
        m.total_turns = int(d.get("total_turns", 0))
        m.last_session = str(d.get("last_session", ""))
        m.voice = str(d.get("voice", ""))
        return m


class SubjectiveHemisphere:
    def __init__(self, cfg, bus, ltm):
        self.cfg, self.bus, self.ltm = cfg, bus, ltm
        self.model = UserModel()
        self._session_apps: Counter = Counter()
        self._loaded = False

    # -- persistence --------------------------------------------------------
    def load(self) -> None:
        data = self.ltm.load_state("preferences", {})
        if data:
            self.model = UserModel.from_json(data)
        self.model.sessions += 1
        self._loaded = True
        self.bus.log(
            f"user model: {self.model.sessions} sessions, "
            f"verbosity {self.model.verbosity:.2f}", "memory")

    def save(self) -> None:
        self.model.last_session = datetime.now().isoformat(timespec="seconds")
        self.ltm.save_state("preferences", self.model.to_json())

    @property
    def returning(self) -> bool:
        return self.model.sessions > 1

    # -- learning -----------------------------------------------------------
    def observe(self, intent: Intent, text: str, speaker=None) -> None:
        """Update the model from one user turn. Nudges only, never jumps."""
        self.model.total_turns += 1
        hour = str(datetime.now().hour)
        self.model.hour_counts[hour] = self.model.hour_counts.get(hour, 0) + 1

        if _SHORTER.search(text):
            self.model.verbosity = max(0.0, self.model.verbosity - 0.34)
            self.bus.log(f"verbosity -> {self.model.verbosity:.2f}", "memory")
        elif _LONGER.search(text):
            self.model.verbosity = min(2.0, self.model.verbosity + 0.34)
            self.bus.log(f"verbosity -> {self.model.verbosity:.2f}", "memory")

        if speaker is not None:
            if _SLOWER.search(text):
                speaker.speed = max(0.80, speaker.speed - 0.06)
                self.model.speech_rate = speaker.speed
            elif _FASTER.search(text):
                speaker.speed = min(1.35, speaker.speed + 0.06)
                self.model.speech_rate = speaker.speed

        if intent.name == "launch":
            target = intent.slots.get("target", "").strip().lower()
            if target:
                self._session_apps[target] += 1
                self.model.app_counts[target] = self.model.app_counts.get(target, 0) + 1

    def observe_salience(self, stm) -> None:
        for term, score in stm.salient(6):
            if score > 1.2:
                self.model.topic_counts[term] = self.model.topic_counts.get(term, 0) + 1

    # -- framing ------------------------------------------------------------
    def frame(self, intent: Intent, stm) -> StyleFrame:
        slot = self.cfg.timeslot()
        verbosity = int(round(min(2.0, max(0.0, self.model.verbosity))))

        # Late at night, tighten regardless of learned preference.
        if slot == "late_night":
            verbosity = min(verbosity, 1)

        # Length, and the reason these are so much longer than they were.
        #
        # The old budgets were one, three and five sentences, and they were
        # wrong on the operator's actual preference: he does not mind listening,
        # every spoken word ends up in the transcript, and the transcript is
        # what long-term memory is built out of — so a clipped answer costs
        # twice, once in the room and again in the vault. An answer now runs as
        # long as it has something to say and stops when it does not.
        #
        # These stay as budgets rather than being removed because the failure
        # they prevent is real and different: a small model with no ceiling
        # restates its own answer in different words until it hits the token
        # limit. "As long as it needs" is an instruction to stop when finished,
        # not permission to pad.
        budget = {
            0: "two or three sentences — he has asked for brevity",
            1: ("as long as the answer genuinely needs, and no longer. A "
                "sentence or two when that covers it; three or four paragraphs "
                "when the subject has that much in it. Do not compress a real "
                "answer to be polite, and do not restate a finished one"),
            2: ("the full version — several paragraphs, all the specifics, the "
                "reasoning as well as the conclusion. He has asked for depth, "
                "so give him the whole thing in one go rather than offering it"),
        }[verbosity]

        return StyleFrame(
            tone="crisp",
            verbosity=verbosity,
            time_flavor=self.cfg.persona.get("character", {})
                            .get("time_flavor", {}).get(slot, ""),
            proactive=self._proactive(intent, stm),
            sentence_budget=budget,
        )

    def _proactive(self, intent: Intent, stm) -> str | None:
        """Something worth volunteering, or nothing.

        Held to a high bar on purpose. An assistant that offers a suggestion every
        turn is a nuisance; one that offers a genuinely useful one occasionally is
        the entire point. Currently fires only on a strong, specific signal.
        """
        if intent.name != "launch" or self.model.sessions < 3:
            return None
        target = intent.slots.get("target", "").lower()
        if not target:
            return None
        pairs = [t for t, c in self.model.app_counts.items() if c >= 4 and t != target]
        if len(pairs) >= 2 and self._session_apps[target] == 1:
            return None    # signal exists but is not specific enough to be useful
        return None

    # -- prompt assembly ----------------------------------------------------
    def system_prompt(self, facts: dict, style: StyleFrame, stm,
                      directive: str = "", calibration: str = "") -> str:
        """Build the model's system prompt: persona, ground truth, style.

        `directive` is the curiosity layer's instruction for this turn — either a
        verified recall from the vault or a request for one follow-up question.
        It goes last, after the length budget, because it is the one instruction
        that is allowed to cost a sentence.

        `calibration` is the VOSS or HackBot spine, and it goes *after* HIVE's
        own and never instead of it. Both profiles were written for a text chat
        where a code block is a reasonable thing to emit; HIVE's spine is what
        establishes that everything here is spoken out loud, and a calibration
        that replaced it would produce an answer that reads well and cannot be
        listened to.
        """
        spine = self.cfg.persona.get("character", {}).get("spine", "").strip()

        fact_lines = "\n".join(f"  - {k.replace('_', ' ')}: {v}" for k, v in facts.items())
        parts = [
            spine,
            "",
            calibration.strip(),
            "",
            "GROUND TRUTH — these are verified. Never contradict them, never guess "
            "at anything they already answer:",
            fact_lines,
            "",
            f"LENGTH: {style.sentence_budget}.",
            "",
            # The delivery rules. These are here rather than in the persona
            # spine because they are about the *channel* — anything written for
            # a screen and then spoken has a set of recognisable tells, and
            # these are the ones that survive being told to a small model.
            "SPEAKING, not writing. Everything you produce is read aloud by a "
            "speech synthesiser that has no markup and no emphasis: it can only "
            "hear punctuation.",
            "  - Never number your points. Say \"first\", \"then\", \"and the "
            "last one\" — a numeral is pronounced as the word \"one\", so a "
            "numbered list is spoken as \"one. two. three.\"",
            "  - No bullets, headers, asterisks, code blocks or emoji.",
            "  - Contractions always: it's, that's, you'd, I'll, doesn't.",
            "  - Start a sentence with And, But or So when that is how it would "
            "be said. Never \"furthermore\", \"additionally\", \"in conclusion\", "
            "\"it is important to note\".",
            "  - Keep sentences under about twenty-two words and put a comma "
            "where you would take a breath. A long sentence with no internal "
            "punctuation is spoken in one breathless run.",
            "  - Vary sentence length deliberately. A short one after two long "
            "ones is what makes speech sound like a person thinking.",
            "  - Say numbers, dates and units as words: \"eleven forty\", \"two "
            "gigabytes\", \"the ninth of August\".",
            "  - Lead with the answer, then the reasoning. Someone listening "
            "cannot skim ahead to find the point.",
        ]
        if style.time_flavor:
            parts.append(style.time_flavor)

        salient = [t for t, _ in stm.salient(6)]
        if salient:
            parts.append(f"Live in this conversation: {', '.join(salient)}.")

        if self.model.sessions > 1:
            top = sorted(self.model.topic_counts.items(), key=lambda kv: -kv[1])[:4]
            if top:
                parts.append(
                    "Recurring subjects across past sessions: "
                    + ", ".join(t for t, _ in top) + "."
                )
        if directive:
            parts.append("")
            parts.append(directive)
        return "\n".join(p for p in parts if p is not None)
