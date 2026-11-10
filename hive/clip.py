"""Pasted material — the third input channel.

HIVE has had two ways in since v1: the microphone and the terminal. Both carry a
*request*. Neither carries a *source*, and the difference matters more than it
looks.

    "what does this mean for a pentest"          a request
    four paragraphs of somebody else's writing   a source

Before this module the second one had nowhere to go. Pasting a block into the
`$` prompt and pressing Enter made it a user turn: it went through the command
grammar, landed in short-term memory verbatim, polluted the salience map with
whoever's vocabulary it happened to be, and got folded into the vault on the way
out as though Jay had said it. Asking a *second* question about it meant hoping
it was still inside the compaction window.

So a paste is held here instead, as a clipping, and handed to the model as a
quoted source alongside every turn until it is dropped. Three properties follow
from that and all three are the point:

  it survives the turn      Ask four questions about one paste. That is the
                            actual workflow — the first question is never the
                            only one.
  it is not you             It goes in as a separate system message that says
                            where it came from, so the model cannot mistake a
                            quoted claim for something Jay asserted, and HIVE
                            cannot repeat it back as its own knowledge.
  it is visible             A chip above the input says what is held, because
                            context you cannot see is context you will forget is
                            there — and a stale clipping silently steering an
                            unrelated answer is the exact failure this would
                            otherwise introduce.

Nothing here is written to disk. A clipping dies with the session like the rest
of short-term memory, and unlike a turn it is never consolidated to the vault:
it is somebody else's text, and the vault is Jay's record of his own work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def _ago(seconds: float) -> str:
    """Rough, spoken-sounding age. Nobody needs a clipping timed to the second."""
    if seconds < 45:
        return "just now"
    minutes = seconds / 60.0
    if minutes < 1.5:
        return "a minute ago"
    if minutes < 55:
        return f"{minutes:.0f} minutes ago"
    hours = minutes / 60.0
    return "an hour ago" if hours < 1.5 else f"{hours:.0f} hours ago"


@dataclass
class Clipping:
    """One block of pasted text, and what HIVE is allowed to say about it."""

    text: str
    source: str = "clipboard"
    ts: float = field(default_factory=time.time)
    # Words cut off the end to fit the per-clipping cap. Carried rather than
    # discarded because the model has to be told what it has *not* seen —
    # answering "the text doesn't cover that" about a passage that was silently
    # truncated is a confident wrong answer wearing the costume of a careful one.
    dropped: int = 0

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def lines(self) -> int:
        return self.text.count("\n") + 1

    @property
    def label(self) -> str:
        """The opening words, for the pane and the chip. Enough to recognise it."""
        head = " ".join(self.text.split()[:7])
        return head + ("…" if len(self.text.split()) > 7 else "")

    def age(self) -> str:
        return _ago(time.time() - self.ts)


class Clippings:
    """Everything pasted in this session that has not been dropped yet."""

    def __init__(self, cfg, bus=None):
        self.cfg, self.bus = cfg, bus
        self.max_held = int(cfg.get("clip.max_held", 3))
        self.max_chars = int(cfg.get("clip.max_chars", 6000))
        self.max_context = int(cfg.get("clip.max_context_chars", 12000))
        self.min_words = int(cfg.get("clip.min_words", 40))
        self._held: list[Clipping] = []
        # Counted for the session note, which records that the conversation
        # leaned on pasted material without reproducing somebody else's text.
        self.pasted = 0

    # -- what counts as a clipping -------------------------------------------
    def is_block(self, text: str) -> bool:
        """Does this paste belong in the line, or held as a source?

        A line break is decisive: nothing you would type at a `$` prompt has one
        in it, and the input row is one line tall by design.

        Length alone is not, and that asymmetry is deliberate. A tracking URL
        runs to three hundred characters and is exactly the kind of thing you
        paste *into* a request; a paragraph lifted out of a PDF often arrives as
        one unbroken line. Word count separates them — a URL is one word.
        """
        text = text or ""
        return "\n" in text.strip() or len(text.split()) >= self.min_words

    # -- holding --------------------------------------------------------------
    def hold(self, text: str, source: str = "clipboard") -> Clipping | None:
        """Take a block in. Returns the clipping, or None if there was nothing."""
        text = (text or "").strip()
        if not text:
            return None

        # Pasting the same thing twice is a double keystroke, not a second
        # source. Refresh it and move it to the front rather than holding two
        # copies and spending the context budget on both.
        for existing in self._held:
            if existing.text == text:
                existing.ts = time.time()
                self._held.remove(existing)
                self._held.append(existing)
                return existing

        dropped = 0
        if len(text) > self.max_chars:
            kept = text[: self.max_chars].rsplit(" ", 1)[0]
            dropped = len(text.split()) - len(kept.split())
            text = kept

        clip = Clipping(text=text, source=source, dropped=dropped)
        self._held.append(clip)
        self.pasted += 1
        self._trim()
        if self.bus is not None:
            self.bus.log(f"held {clip.words} words from the {source}", "clip")
        return clip

    def _trim(self) -> None:
        """Oldest out first, on both the count cap and the context budget."""
        while len(self._held) > self.max_held:
            self._held.pop(0)
        while len(self._held) > 1 and sum(len(c.text) for c in self._held) > self.max_context:
            self._held.pop(0)

    def drop(self) -> str:
        """Let go of the newest. The undo for a paste you did not mean."""
        if not self._held:
            return "nothing is held."
        clip = self._held.pop()
        return f"dropped [{len(self._held) + 1}] — {clip.words} words. {self.status()}"

    def clear(self) -> str:
        if not self._held:
            return "nothing is held."
        n = len(self._held)
        self._held.clear()
        return f"dropped {n} clipping(s). Nothing is held now."

    # -- reading --------------------------------------------------------------
    @property
    def held(self) -> list[Clipping]:
        return list(self._held)

    def __len__(self) -> int:
        return len(self._held)

    @property
    def words(self) -> int:
        return sum(c.words for c in self._held)

    def status(self) -> str:
        if not self._held:
            return "Nothing held."
        return f"{len(self._held)} held, {self.words} words."

    def chip(self) -> str:
        """The one line the UI shows above the input while anything is held."""
        if not self._held:
            return ""
        if len(self._held) == 1:
            clip = self._held[0]
            return f"HELD  {clip.words}w  {clip.label}"
        return f"HELD  {len(self._held)} clippings  {self.words}w"

    def summary(self) -> str:
        """The `clip` command's answer."""
        if not self._held:
            return ("Nothing held. Paste a block into the prompt, or copy it and "
                    "type `paste`, and HIVE will hold it as a source you can ask "
                    "about.")
        out = [f"{len(self._held)} clipping(s) held, {self.words} words total:"]
        for i, clip in enumerate(self._held, 1):
            line = (f"  [{i}] {clip.words} words · {clip.lines} line(s) · "
                    f"{clip.age()}")
            if clip.dropped:
                line += f" · {clip.dropped} words cut to fit"
            out.append(line)
            out.append(f"      {clip.label}")
        out.append("")
        out.append("Ask about them in plain language — they go to HIVE with every "
                   "turn until you drop them.")
        out.append("`clip drop` lets go of the newest · `clip clear` lets go of all "
                   "· `clip show N` prints one back")
        return "\n".join(out)

    def show(self, index: int) -> str:
        if not 1 <= index <= len(self._held):
            return f"there is no clipping [{index}]. {self.status()}"
        clip = self._held[index - 1]
        head = f"[{index}] {clip.words} words, pasted {clip.age()}"
        if clip.dropped:
            head += f" ({clip.dropped} words were cut to fit)"
        return f"{head}\n{clip.text}"

    # -- the model's view ------------------------------------------------------
    def context_block(self) -> str:
        """The system message the held clippings become. '' when nothing is held.

        Written as an instruction about *provenance* rather than a bare dump. A
        small model handed unattributed text at the top of its context treats it
        as its own knowledge and will happily assert it back — which is how a
        quoted claim from a forum post becomes something "HIVE said". Saying
        whose words these are, and that not knowing is an allowed answer, is
        cheaper and far more reliable than catching it downstream.
        """
        if not self._held:
            return ""
        parts = [
            "PASTED MATERIAL. The operator copied the following in from "
            "somewhere else and is asking you about it. It is a quoted source: "
            "not something you said, and not something he told you. Do not "
            "assert its claims as your own knowledge — attribute them to the "
            "text when it matters.",
            "Answer from what is actually in it. If it does not cover what he "
            "asked, say so plainly and then say what you do know; being told "
            "where the text stops is the reason he pasted it rather than asking "
            "cold.",
            "This is being read aloud, so never recite it back verbatim unless "
            "he asks for that. Summarise, quote the one phrase that matters, "
            "answer the question.",
        ]
        for i, clip in enumerate(self._held, 1):
            head = f"[{i}] {clip.words} words, pasted {clip.age()}"
            if clip.dropped:
                head += (f" — TRUNCATED: the last {clip.dropped} words are not "
                         f"here and you have not seen them")
            parts.append(f"{head}\n<<<\n{clip.text}\n>>>")
        return "\n\n".join(parts)
