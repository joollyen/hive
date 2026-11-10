"""Prosody — the layer between what HIVE decides to say and what Kokoro renders.

Kokoro has no SSML, no emotion tags and no emphasis markup. It ignores capitals.
Everything it knows about rhythm, pitch movement and pausing it reads out of one
channel: **punctuation**. Its vocabulary accepts exactly these marks and silently
deletes everything else before the model ever sees the text —

    !  "  (  )  ,  .  :  ;  ?  —  “  ”  …

which has three consequences that this module exists to act on.

1. **A hyphen is not a dash.** `-` is not in that set, so " - " between two
   clauses is removed outright: the pause a writer intended simply does not
   happen and the two clauses run together. `—` (U+2014) *is* in the set and
   produces a real parenthetical break. Language models write the first one
   almost exclusively.

2. **Document structure is read aloud as words.** A model that answers with
   "1. Linguistic modelling. 2. Emotional intelligence." is phonemised to
   `wˈɒn. ... tˈuː.` — HIVE says "One. Linguistic modelling. Two. Emotional
   intelligence." Verified against this machine's own Kokoro build, and heard in
   a real session. People do not speak in numbered lists; they say "first",
   "then", "and finally". Enumeration has to be rewritten, not stripped: strip
   the marker and the listener loses the structure, keep it and they hear a
   document being read at them.

3. **The pause has to be in the text or it does not exist.** Kokoro trims
   silence off both ends of every unit it renders, so the only pauses that
   survive are the ones punctuation produces inside a unit and the ones the
   joiner inserts between units. A twenty-five word sentence with no internal
   comma is rendered as one unbroken breath, which is the single most
   recognisable "read by a machine" artefact there is.

The countermeasure for (3) is the reason `breathe()` exists. Speakers phrase into
intonational units of roughly one and a half to three seconds — about fifteen to
twenty-five words — bounded by a pause, a pitch reset and pre-boundary
lengthening. Prose written to be read silently ignores that budget constantly.
So a clause that overruns it gets a boundary inserted at the best *grammatical*
seam available, which is what a person would have done with their lungs.

Everything here is a pure, deterministic function of the text. That is load
bearing rather than tidiness: audio is cached by the string that produced it, so
anything that varied per utterance — a random discourse marker, a shuffled
filler — would miss the cache every single time and pay full synthesis latency
on lines that are supposed to be free. Variety comes from what is being said,
not from a dice roll about how to say it.
"""

from __future__ import annotations

import re

# --- what survives phonemisation -------------------------------------------
# Kokoro's vocabulary, restricted to the marks that carry prosody. Anything
# outside this is either deleted before synthesis or read out as a word, so this
# set is the entire palette available for shaping delivery.
KEPT = set(".,;:!?…—()\"“”")

# --- structure that must not be spoken as words ----------------------------

_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>[^\n]*?)\s*$", re.M)
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC = re.compile(r"(?<![\w*])\*([^*\n]{1,80})\*(?![\w*])")
_CODE = re.compile(r"`([^`\n]{1,120})`")
_LINK = re.compile(r"\[([^\]\n]{1,80})\]\([^)\s]+\)")
_RULE = re.compile(r"^\s*([-*_]\s*){3,}$", re.M)

# A bullet at the head of a line. Replaced with a sentence boundary rather than
# deleted: the bullets were carrying the separation between items, and removing
# them without putting anything back runs every item into one long clause.
_BULLET = re.compile(r"^[ \t]*[-*•·–—]\s+", re.M)

# An enumerator — "1." / "2)" / "3 -" at the start of a clause. The negative
# lookahead for a digit keeps "1.4 million" and "in 2026. 1.4x" intact, and the
# lookbehind restricts matching to clause-initial position so a decimal or a
# version number mid-sentence is never touched.
_ENUM = re.compile(
    r"(?:(?<=^)|(?<=[.:;!?…]\s)|(?<=[.:;!?…]\n)|(?<=\n))"
    r"[ \t]*(?P<n>[1-9])[.)][ \t]+(?!\d)")

_ORDINAL = {
    1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth",
    6: "Sixth", 7: "Seventh", 8: "Eighth", 9: "Ninth",
}

# --- dashes ----------------------------------------------------------------
# Every one of these is deleted outright by the vocabulary filter, taking its
# pause with it. Mapped onto the em dash, which is kept.
_DASH = re.compile(r"\s+[-–—]{1,3}\s+|\s+--\s+")
_ELLIPSIS = re.compile(r"\.\s?\.\s?\.+")

# --- things Kokoro reads wrongly -------------------------------------------
#
# This list is short on purpose, and it used to be much longer. espeak already
# spells an unknown capitalised run out letter by letter, and it does it *better*
# than a hand-written expansion does, because it stresses the run as one word:
#
#     CPU   -> sˌiːpˌiːjˈuː      which is "see-pee-YOU", how a person says it
#     C P U -> sˈiː pˈiː jˈuː    three separate stresses, which is a machine
#
# Measured that way across every acronym HIVE handles, the expansions for CPU,
# GPU, URL, API, SSH, TTS, ASR, LLM, AI, eJPT, eCPPT, CRTO, SMB, NTLM and LDAP
# were all making the delivery worse than doing nothing, and "LDAP" -> "L DAP"
# was turning a word people say ("el-dap") into five letters nobody says.
#
# What survives is only the cases where espeak is genuinely wrong: units, which
# it reads as letters; the two acronyms it mangles into a syllable; and the ones
# with a local meaning it cannot know.
_SPOKEN = [
    (re.compile(r"\bGB\b"), "gigabytes"), (re.compile(r"\bMB\b"), "megabytes"),
    (re.compile(r"\bKB\b"), "kilobytes"), (re.compile(r"\bTB\b"), "terabytes"),
    (re.compile(r"\bms\b"), "milliseconds"),
    (re.compile(r"\bRAM\b"), "ram"),          # espeak: "ar-ay-em"
    (re.compile(r"\bVRAM\b"), "V ram"),       # espeak: "vram", one word
    (re.compile(r"\bOSCP\b", re.I), "O S C P"),   # espeak: 'ˈɒskp'
    (re.compile(r"\bSQL\b"), "sequel"),
    (re.compile(r"\bCPP\b"), "Cal Poly Pomona"),
    # CompTIA's certifications. "A plus" is read as "uh plus"; hyphenating it
    # gets the letter name back without gluing the two words together, which is
    # what happens to "Network-plus".
    (re.compile(r"\bA\s+plus\b"), "A-plus"),
    (re.compile(r"\bvs\.?\b", re.I), "versus"),
    (re.compile(r"&"), " and "),
    # Currency and percent. Both symbols are outside KEPT, so they are deleted
    # silently and "$129.99" is spoken as "one hundred and twenty nine point
    # nine nine" — a number with no unit, which is the one form of a price that
    # tells the listener nothing. Only relevant since HIVE went online: a
    # shopping search is the first thing in the system that reads a price aloud.
    # Cents are dropped when they are ".00" and spoken bare otherwise, which is
    # how a person says it: "one twenty nine ninety nine", not "point nine nine".
    (re.compile(r"\$\s?(\d[\d,]*)\.00\b"), r"\1 dollars"),
    (re.compile(r"\$\s?(\d[\d,]*)\.(\d{2})\b"), r"\1 dollars \2"),
    (re.compile(r"\$\s?(\d[\d,]*)"), r"\1 dollars"),
    (re.compile(r"(\d)\s?%"), r"\1 percent"),
    # "read/write" -> "read or write", but only between two real words. An
    # unrestricted rule turns "24/7" into "twenty four or seven" and a date into
    # nonsense, which is a worse failure than the one it fixes.
    (re.compile(r"\b([a-zA-Z]{3,})\s*/\s*([a-zA-Z]{3,})\b"), r"\1 or \2"),
]

# Expanded during repunctuation rather than at synthesis time, because the full
# stop inside an abbreviation looks like the end of a sentence to everything
# downstream — the recapitaliser was upper-casing the word after "i.e.", and the
# splitter would happily start a new prosodic unit there.
_ABBREV = [
    # An abbreviated month, which is broken twice over and both halves are fixed
    # here. espeak phonemises "Sept." as /sˈɛpt/ — the word "sept", rhyming with
    # "kept", not the month — and the full stop inside it reads as the end of a
    # sentence to the unit splitter below, so "on Wednesday, Sept. 23rd" is
    # delivered as "on Wednesday, September." [150ms, falling contour]
    # "Twenty-third." Measured through this machine's own espeak build.
    #
    # "May" is absent because it is never abbreviated, and "Sept" carries an
    # optional fourth letter because both spellings are written. Guarded by a
    # lookahead for a following day number so a surname, a variable or
    # "March" as a verb is never rewritten. The ordinal suffix has to be part of
    # that lookahead rather than left to \b — "23rd" has no word boundary between
    # the digits and the "rd", so a bare \d{1,2}\b matches the plain "23" and
    # misses every written date that spells its ordinal out.
    (re.compile(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.\s*"
                r"(?=\d{1,2}(?:st|nd|rd|th)?\b)"),
     lambda m: {"Jan": "January", "Feb": "February", "Mar": "March",
                "Apr": "April", "Jun": "June", "Jul": "July",
                "Aug": "August", "Sep": "September", "Sept": "September",
                "Oct": "October", "Nov": "November",
                "Dec": "December"}[m.group(1)] + " "),
    (re.compile(r"\bi\.e\.\s*", re.I), "that is, "),
    (re.compile(r"\be\.g\.\s*", re.I), "for example, "),
    (re.compile(r"\betc\.(?=\s|$)", re.I), "and so on"),
]

# --- breath groups ---------------------------------------------------------
# Speakers phrase into units of roughly 1.6–3 seconds, which at conversational
# rate is about fifteen to twenty-five words, bounded by a pause and a pitch
# reset. Past the ceiling the delivery audibly runs out of air; below the floor a
# fragment gets its own falling contour and sounds like an afterthought.
BREATH_MAX_WORDS = 22
BREATH_MIN_WORDS = 5

# Where a breath may be taken, best first. A subordinator is the strongest seam
# in English prose — the clause boundary is already there, it just has no
# punctuation marking it. A bare coordinator is weaker, and a relative pronoun
# weaker still, because "the thing, that I meant" is wrong often enough to be
# worth ranking last.
_BREATH_SEAMS = [
    (3, re.compile(r"\s+(?=(?:because|so that|which means|even though|although|"
                   r"whereas|unless|whether|since|while|though|until|before|"
                   r"after|once|if)\b)", re.I)),
    (2, re.compile(r"\s+(?=(?:and then|but|and yet|or else|so|yet)\b)", re.I)),
    (2, re.compile(r"\s+(?=(?:with|without|against|between|across|through)\b)", re.I)),
    (1, re.compile(r"\s+(?=(?:that|which|where|when|who)\b)", re.I)),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_WS = re.compile(r"\s+")

# Punctuation runs that a rewrite can produce and no reader would ever write:
# "modelling,." from a bullet that already ended in a comma, "?." from a
# question inside a list item.
_DOUBLED = re.compile(r"([,;:—…])\s*([.,;:!?…])")
_SPACE_BEFORE = re.compile(r"\s+([,.;:!?…])")
_ORPHAN = re.compile(r"(?:^|(?<=[.!?…]\s))\s*([,;:—])\s*")
# Runs the ellipsis rule has already had its chance at, so anything left is an
# artefact rather than a mark someone meant.
_REPEATED_STOP = re.compile(r"([.!?])\1+|([.!?])\s+\2")

# A sentence start that a rewrite created. Bullets and headers arrive without
# terminal punctuation, so turning them into sentences leaves the next one
# opening on a lowercase word — which Kokoro renders with a continuation
# contour, as though the previous sentence had not finished.
_SENTENCE_START = re.compile(r"(^|[.!?…]\s+)([a-z])")


def _words(text: str) -> int:
    return len(text.split())


# --- stage 1: structure -----------------------------------------------------

def _enumerate_to_prose(text: str) -> str:
    """"1. X  2. Y" -> "First, X  Second, Y".

    Rewritten rather than deleted. The numbers were carrying the shape of the
    answer, and a listener who loses them cannot tell where one item stops and
    the next starts — which is how "1. Linguistic modelling 2. Emotional
    intelligence" collapses into one run-on noun phrase. Ordinals put the shape
    back in a form a person would actually say out loud.

    Idempotent: the output contains no enumerators, so a streaming buffer that
    passes through here repeatedly is only ever transformed once.
    """
    def sub(m: re.Match) -> str:
        return f"{_ORDINAL[int(m.group('n'))]}, "

    return _ENUM.sub(sub, text)


def _bullets_to_sentences(text: str) -> str:
    """A bullet is a boundary. Replace it with one rather than dropping it."""
    def sub(m: re.Match) -> str:
        start = m.start()
        # Whatever preceded this bullet already ended the previous item; only add
        # a stop if it did not end in one.
        head = text[:start].rstrip()
        if not head or head[-1] in ".!?…:;":
            return ""
        return ". " if head[-1] not in ",—" else " "

    return _BULLET.sub(sub, text)


def _header_to_sentence(m: re.Match) -> str:
    """A header is a spoken sentence, not a silent label.

    Dropping the hashes and nothing else leaves the title fused to the paragraph
    below it — "Contextual awareness Training models to understand" — because a
    header carries its boundary in the layout rather than in punctuation.
    """
    title = m.group("title").strip().rstrip(".:")
    return f"{title}." if title else ""


def despatialise(text: str) -> str:
    """Remove everything that is layout rather than language."""
    text = _RULE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _ITALIC.sub(r"\1", text)
    text = _HEADER.sub(_header_to_sentence, text)
    text = _bullets_to_sentences(text)
    text = _enumerate_to_prose(text)
    return text


def recapitalise(text: str) -> str:
    """Sentence-case the openings that this module's own rewrites created."""
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)


# --- stage 2: marks Kokoro can actually use --------------------------------

def repunctuate(text: str) -> str:
    """Map written punctuation onto the marks that survive phonemisation.

    The hyphen is the important one. Models write " - " for a parenthetical
    break constantly, and Kokoro deletes it, so the break the writer heard in
    their head never reaches the listener. The em dash is kept by the vocabulary
    and produces the pause that was intended.
    """
    for pat, rep in _ABBREV:
        text = pat.sub(rep, text)
    text = _ELLIPSIS.sub("…", text)
    text = _DASH.sub(" — ", text)
    # A doubled stop, which a model produces whenever it ends a sentence on an
    # abbreviation or a quotation. Kokoro renders "job.." as two boundaries and
    # the second one is a beat of silence with nothing on either side of it.
    text = _REPEATED_STOP.sub(lambda m: m.group(1) or m.group(2), text)
    # A colon introducing a list of clauses reads better as anticipation than as
    # a full stop, and Kokoro renders it that way — but only if it is not
    # immediately followed by another mark.
    text = _DOUBLED.sub(r"\2", text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _ORPHAN.sub(" ", text)
    return text


def to_spoken(text: str) -> str:
    """Expand what Kokoro would mispronounce. Cheap, and it runs on every unit."""
    for pat, rep in _SPOKEN:
        text = pat.sub(rep, text)
    return _WS.sub(" ", text).strip()


# --- stage 3: breath groups -------------------------------------------------

def _best_breath(clause: str) -> int:
    """Index of the best place to take a breath in `clause`, or -1.

    Preference is grammatical quality first, then proximity to the middle. The
    middle matters because the point is to produce two speakable phrases: a seam
    three words from the end leaves a fragment, which sounds worse than the long
    clause it was meant to fix.
    """
    words = clause.split()
    if len(words) < BREATH_MIN_WORDS * 2:
        return -1
    mid = len(clause) / 2.0
    best = (-1, 1e9, -1)                    # (priority, distance from middle, at)
    for pri, pat in _BREATH_SEAMS:
        for m in pat.finditer(clause):
            at = m.end()
            head, tail = clause[:at], clause[at:]
            if _words(head) < BREATH_MIN_WORDS or _words(tail) < BREATH_MIN_WORDS:
                continue
            dist = abs(at - mid)
            if pri > best[0] or (pri == best[0] and dist < best[1]):
                best = (pri, dist, at)
    return best[2]


def breathe(text: str, max_words: int = BREATH_MAX_WORDS) -> str:
    """Insert breath boundaries into clauses that run past a speakable length.

    Operates per punctuation-delimited clause, not per sentence: a sentence with
    three commas in it is already phrased and needs nothing. Only the runs that
    have no internal boundary at all get one, and only where the grammar offers
    a seam — an arbitrary break at word twenty-two would land mid-phrase and
    sound worse than the breathlessness it replaced.
    """
    out: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        pieces: list[str] = []
        # Keep the marks: they are the existing phrasing and must survive intact.
        for chunk in re.split(r"(?<=[,;:—])\s+", sentence):
            while _words(chunk) > max_words:
                at = _best_breath(chunk)
                if at < 0:
                    break
                head, tail = chunk[:at].rstrip(), chunk[at:].lstrip()
                if not head or not tail:
                    break
                # A comma is the least intrusive boundary that Kokoro renders as
                # a pitch continuation rather than a fall — which is what a
                # breath inside a sentence is supposed to be.
                pieces.append(head if head[-1] in ",;:—" else head + ",")
                chunk = tail
            pieces.append(chunk)
        out.append(" ".join(p for p in pieces if p))
    return " ".join(out)


def terminate(text: str) -> str:
    """Guarantee a final mark so the last phrase actually lands.

    A unit ending on a bare word is given a continuation contour: it trails off
    as though more is coming, and then nothing does. That reads as HIVE being
    cut off mid-sentence, which is the impression a truncated answer gives even
    when the answer was complete.
    """
    t = text.rstrip()
    if not t:
        return t
    # The em dash belongs here with the commas and colons: a phrase that ends on
    # one is deliberately mid-utterance. HIVE's own refusal lines are built that
    # way — "I can't do that one —" is a lead that another clause finishes — and
    # closing it produces "I can't do that one —." which is a pause followed by
    # a full stop where a continuation was meant.
    return t if t[-1] in ".!?…:;,—)\"”" else t + "."


# --- the pipeline -----------------------------------------------------------

def shape(text: str, max_words: int = BREATH_MAX_WORDS,
          breath: bool = True, terminal: bool = True) -> str:
    """Everything above, in the order the stages depend on each other.

    Structure first (it creates punctuation), then mark normalisation (it fixes
    what structure produced), then breathing (it needs the final punctuation to
    know what is already phrased), then termination.

    `terminal=False` for a buffer that is still being generated. Closing an
    incomplete utterance is wrong twice over: the half-sentence is given a
    falling contour it has not earned, and — because the streaming path feeds
    the shaped remainder back in as its next buffer — the invented full stop
    persists, so the rest of the sentence arrives as a new one and is spoken as
    a separate thought.
    """
    if not text or not text.strip():
        return ""
    t = despatialise(text)
    t = repunctuate(t)
    t = _WS.sub(" ", t).strip()
    if breath:
        t = breathe(t, max_words)
    return recapitalise(terminate(t) if terminal else t)


# --- delivery: what happens at each seam ------------------------------------
#
# Measured on this machine: mid-sentence marks buy Kokoro only 40–65ms of
# internal pause, and sentence-final marks buy none at all because the trailing
# silence is trimmed off every unit before it is returned. So the rhythm between
# units is almost entirely this table's responsibility, and the values are
# chosen by what the mark *means* rather than by how it looks.
#
# (crossfade ms, pause ms). The crossfades stay short deliberately: they exist to
# hide the discontinuity between two independently-rendered units, not to blend
# them, because consecutive units contain different words and a long overlap
# slurs them together.
SEAM = {
    ".":  (0, 150),     # a statement lands, and the contour resets
    "!":  (0, 140),
    "?":  (0, 185),     # a question leaves room for the answer that is not coming
    "…":  (0, 215),     # this mark *is* the trailing pause; give it the time
    ":":  (6, 118),     # anticipation — something is being introduced
    ";":  (8, 104),
    "—":  (12, 74),     # parenthetical: lighter than a clause, heavier than none
    ",":  (12, 26),     # the comma already carries most of its own pause
    "":   (16, 0),      # a break forced by length. Should be undetectable.
}

# A speaker takes a beat before pivoting. These are the words that mark a pivot,
# and the extra silence in front of one is the difference between a new thought
# and a longer sentence.
_PIVOT = re.compile(
    r"^\s*(?:but|so|still|though|however|that said|then again|either way|"
    r"mind you|the thing is|which is why|and yet|actually|honestly)\b", re.I)
PIVOT_PAUSE_MS = 45


def seam_for(unit: str, next_unit: str = "") -> tuple[float, float]:
    """Crossfade and pause to apply after `unit`, given what follows it."""
    tail = unit.rstrip()[-1:] if unit.strip() else ""
    over, pause = SEAM.get(tail, SEAM[""])
    if pause and next_unit and _PIVOT.match(next_unit):
        pause += PIVOT_PAUSE_MS
    return float(over), float(pause)


# --- delivery: rate ---------------------------------------------------------
#
# Phrase-final lengthening is the best-attested boundary cue in speech: the
# material before a prosodic boundary is stretched by roughly a fifth. Utterance
# onsets are slower too — that is where a listener orients — and continuations
# inside an already-established sentence run slightly quicker, which is the
# tempo half of declination.
#
# The adjustments are deliberately tiny and hard-bounded. Anything audible as a
# speed change is worse than a flat delivery, because a flat delivery merely
# sounds synthetic while a wandering one sounds broken.
RATE_SPAN = 0.05


_POSITION_ADJ = {
    "only":  0.00,      # nothing to shape against
    "open": -0.40,      # orientation: open a shade slower
    "mid":  +0.25,      # continuation: tempo drifts up through an utterance
    "close": -0.30,     # phrase-final lengthening
}


def unit_rate(base: float, position: str, unit: str,
              span: float = RATE_SPAN) -> float:
    """Speaking rate for one unit, given where it sits in the utterance.

    Position rather than index because the streamed path does not know how many
    units there will be until it has spoken most of them — it only ever knows
    "this is the first", "this is more of the same" and "this is the last".

    A single-unit utterance is returned at exactly `base`, and that is load
    bearing rather than a tidy special case: every fixed persona line is one
    unit and they are pre-rendered into the phrase cache at the base rate.
    Perturbing them would miss that cache on every acknowledgement HIVE speaks.
    """
    adj = _POSITION_ADJ.get(position, 0.0)
    if span <= 0 or (position == "only" and not adj):
        return base
    tail = unit.rstrip()[-1:] if unit.strip() else ""
    if tail == "?":
        adj -= 0.35                     # a rise needs room to be heard as one
    elif tail == "…":
        adj -= 0.25
    elif tail == ",":
        adj += 0.15                     # mid-clause, still moving
    if position == "only":
        return base
    rate = base * (1.0 + max(-1.0, min(1.0, adj)) * span)
    return round(max(0.5, min(2.0, rate)), 3)
