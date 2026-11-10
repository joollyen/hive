"""Turning a request back into words — acknowledgements and holding lines.

Two problems with the same root, both raised by the operator:

  "Opening it now."      — for every application, so it says nothing.
  "Let me look."         — for every slow turn, so it says nothing.

Both are drawn from fixed banks, which is why they can be pre-synthesized and
cost zero latency, and also why they are interchangeable across every request
HIVE will ever receive. A line that would fit any question is a line that
answers none of them, and after the fiftieth time it reads as a progress bar
with a voice.

The fix is not more banks. It is to build the line out of the request:

    "open spotify"                             -> "Opening Spotify."
    "what are the differences between          -> "Differences between lorazepam
     lorazepam and propranolol"                    and propranolol, coming up."

The subject extraction here is deliberately conservative. A holding line is
spoken in the first half-second of a turn, before HIVE has understood anything,
so a mangled one is the first thing heard and it is heard often. Every path
falls back to the generic bank rather than emitting something ungrammatical:
"one moment" is boring, and boring beats broken.

Three question shapes, because they want different treatment:

  lookup        "what/how/why/explain/compare X" — has a subject worth naming,
                and naming it is what makes the wait feel like work in progress
                rather than a pause.
  deliberation  "should I / is it worth / what do you think" — the subject is a
                decision, and reading a decision back as a noun phrase produces
                "should i take the ejpt before the oscp, coming up." Generic is
                genuinely the right answer here, and the operator said so:
                thinking out loud is appropriate when there is thinking to do.
  other         anything that survives neither, which falls back.
"""

from __future__ import annotations

import re
import zlib

# Interrogative and imperative scaffolding. What is left after this is the thing
# being asked about. Ordered longest-first inside each group so "what are the"
# wins over "what are".
_LEAD = re.compile(
    r"^(?:hive[,\s]+)?(?:so[,\s]+)?"
    r"(?:tell me about|tell me|walk me through|break down|talk me through|"
    r"give me a rundown on|give me|show me|look up|look into|find out about|"
    r"find out|research|remind me about|remind me|"
    r"i want to know about|i want to know|i would like to know|do you know|"
    r"explain to me|explain|describe|define|compare|contrast|summari[sz]e|"
    r"what are the|what is the|what are|what is|whats|"
    r"who are the|who is the|who are|who is|"
    r"how do you|how do i|how does|how do|how did|how is|how are|how would|"
    r"how can i|how can|how much|how many|"
    r"why does|why do|why is|why are|why did|"
    r"when is|when do|when does|when did|where is|where do|where does|"
    r"which is|which are|which one)"
    r"\s+", re.I)

# A decision being asked about, not a subject being looked up.
_DELIBERATION = re.compile(
    r"^(?:hive[,\s]+)?(?:should|shall|do you think|what do you think|"
    r"is it worth|would it be worth|am i|are you sure|do i need|"
    r"would you|which should|what would you|what does that say|"
    # "why do i keep putting that off" is a question about himself, not a
    # subject to look up. "how do i enumerate SMB shares" is the opposite, so
    # the pronoun is doing the work here and only after why/what.
    r"why (?:do|am|did|does|would) i|what makes me)\b", re.I)

_ARTICLE = re.compile(r"^(?:the|a|an|my|your|some|any)\s+", re.I)
_TRAILING = re.compile(
    r"\s+(?:please|for me|thanks|thank you|exactly|though|then|or not|"
    r"or something|or anything|really)\W*$", re.I)
# A phrase that ends on one of these is a fragment; cut back to before it.
_DANGLING = {
    "and", "or", "but", "with", "for", "to", "of", "in", "on", "at", "by",
    "from", "about", "between", "against", "into", "than", "that", "the", "a",
    "an", "is", "are", "was", "were", "do", "does", "did", "if", "when", "as",
}

MIN_SUBJECT = 6
MAX_SUBJECT_WORDS = 10
MAX_SUBJECT_CHARS = 66

# Some interrogative frames must be kept, in rewritten form, or what is left is
# a bare verb phrase and every template turns ungrammatical: "how do i get
# Proton to launch" reduces to "get Proton to launch", and "Looking at get
# Proton to launch now" is not a sentence. Rewriting to "how to get Proton to
# launch" keeps it a noun phrase and reads better than the original did.
_REFRAME = [
    (re.compile(r"^how (?:do|can|would|should) (?:i|you|we)\s+", re.I), "how to "),
]

# ...and some must be kept whole. "How does the BND4 container store its
# checksums" cannot be reduced at all: strip the auxiliary and the verb stops
# agreeing ("how the BND4 container store"), strip the interrogative and it
# becomes a statement. These questions already read as their own subject, so
# they are used verbatim.
_KEEP_WHOLE = re.compile(
    r"^(?:how|why|when|where|which)\s+(?:do|does|did|is|are|was|were|can|could|"
    r"would|should|will)\b", re.I)


def question_shape(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "other"
    if _DELIBERATION.match(t):
        return "deliberation"
    if _LEAD.match(t):
        return "lookup"
    return "other"


def subject_of(text: str) -> str:
    """The thing being asked about, or '' when nothing safe can be extracted.

    Works on the normalised utterance — already lowercased, contraction-expanded
    and filler-stripped by intents.normalize — so this only has to remove the
    interrogative frame and whatever is left dangling once it is gone.
    """
    t = _WS(text or "")
    if not t:
        return ""
    prefix = ""
    for pat, replacement in _REFRAME:
        m = pat.match(t)
        if m:
            prefix, t = replacement, t[m.end():]
            break
    if not prefix and _KEEP_WHOLE.match(t):
        pass                    # already its own subject; take it verbatim
    elif not prefix:
        lead = _LEAD.match(t)
        if not lead:
            return ""
        t = t[lead.end():]
        t = _ARTICLE.sub("", t)
    t = _TRAILING.sub("", t)
    t = _WS(t)
    if not t:
        return ""

    words = t.split()
    if len(words) > MAX_SUBJECT_WORDS:
        words = words[:MAX_SUBJECT_WORDS]
    # Never end on a preposition or conjunction — "differences between" is worse
    # than no subject at all, because it sounds like HIVE was cut off.
    while words and words[-1].lower() in _DANGLING:
        words.pop()
    if not words:
        return ""

    out = prefix + " ".join(words)
    if len(out) > MAX_SUBJECT_CHARS:
        out = out[:MAX_SUBJECT_CHARS].rsplit(" ", 1)[0]
        while out and out.rsplit(" ", 1)[-1].lower() in _DANGLING:
            out = out.rsplit(" ", 1)[0]
    out = out.strip(" ,.;:-")
    if len(out) < MIN_SUBJECT or " " not in out and len(out) < 9:
        # A single short word is not a subject worth reading back.
        return ""
    return out


def _WS(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def fill(template: str, **slots: object) -> str:
    out = template
    for key, value in slots.items():
        out = out.replace("{" + key + "}", str(value))
    return _WS(out)


def holding_line(cfg, text: str) -> str:
    """The line HIVE says while it is still working. '' means use the bank.

    Returns a tailored line only for lookup-shaped questions with a clean
    subject. Everything else is left to the caller's generic bank, on purpose —
    see the module docstring.
    """
    shape = question_shape(text)
    if shape != "lookup":
        return ""
    subject = subject_of(text)
    if not subject:
        return ""
    # Deterministic, not shuffled — and this is load-bearing rather than a
    # simplification. The line is synthesized speculatively while the request is
    # still being typed, and a shuffle bag hands back a different template when
    # the turn actually runs, so the pre-rendered audio would miss the cache
    # every single time and the whole head start would be thrown away. Variety
    # comes from the subject instead: different questions still draw different
    # templates, and asking the same question twice is a cache hit both times.
    bank = cfg.bank("filler.tailored")
    if not bank:
        return ""
    template = bank[zlib.crc32(subject.encode("utf-8")) % len(bank)]
    return _sentence_case(fill(template, subject=subject))


def proper(name: str) -> str:
    """Present a resolved name the way it is written, not the way it was heard.

    Launcher labels come off a spoken request and arrive lowercase — "spotify",
    "google doc". Title-casing fixes those, and would break the ones that
    already carry their own capitalisation ("ZipRecruiter" -> "Ziprecruiter"),
    so it only fires when there is no casing to lose.
    """
    name = (name or "").strip()
    if not name or any(c.isupper() for c in name):
        return name
    return name.title()


def acknowledgement(cfg, bank: str, title_case: bool = True, **slots: object) -> str:
    """A bank line with the request's own nouns in it, e.g. the app being opened.

    `title_case` is on for names and off for phrases. An application is a proper
    noun and "spotify" heard from the microphone should be said back as
    "Spotify". A search subject is not: `proper()` turns "keyboards like the
    corsair vanguard" into "Keyboards Like The Corsair Vanguard", which reads as
    a product name rather than as a description, and Kokoro ignores capitals so
    it is a defect in the transcript and the vault rather than in the audio.
    """
    template = cfg.line(bank)
    if not template:
        return ""
    if title_case:
        slots = {k: proper(v) if isinstance(v, str) else v for k, v in slots.items()}
    return _sentence_case(fill(template, **slots))
