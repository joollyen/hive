"""Deterministic intent grammar — the fast path.

Anything matched here is executed without ever consulting a language model. That
is worth roughly 300ms per turn, but speed is the lesser reason. The real reason
is that "open Chrome" has exactly one correct interpretation, and routing it
through a probabilistic model introduces a failure mode that regular expressions
simply do not have. The objective hemisphere is supposed to be deterministic; this
is where that property actually comes from.

Everything that does not match falls through to CONVERSE and the model handles it.

Written against how people actually talk to assistants out loud — with filler,
politeness, self-correction and trailing qualifiers — not against clean text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from rapidfuzz import fuzz


class Category(str, Enum):
    """Drives which status word the visualizer shows while working.

    Values must match keys under `status_words` in persona.yaml. Several names
    intentionally alias the same value (LAUNCH/SYSTEM/CONTROL/TIME all read
    "system") — the distinct names document intent at the call site while the
    shared value keeps the vocabulary coherent on screen.
    """

    PRESENCE = "presence"
    LAUNCH = "system"
    VAULT_WRITE = "writing"
    VAULT_READ = "memory"
    SYSTEM = "system"
    CONTROL = "system"
    TIME = "system"
    CONVERSE = "general"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    MEDIA = "system"
    WORKSPACE = "analysis"


@dataclass
class Intent:
    name: str
    category: Category = Category.CONVERSE
    slots: dict = field(default_factory=dict)
    confidence: float = 0.0
    raw: str = ""
    cleaned: str = ""


# --- normalisation ---------------------------------------------------------

_CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "that's": "that is",
    "let's": "let us", "i'm": "i am", "you're": "you are", "don't": "do not",
    "can't": "can not", "won't": "will not", "it's": "it is", "there's": "there is",
    "i've": "i have", "we've": "we have", "how's": "how is",
}

# Politeness and filler that carries no meaning but wrecks naive slot extraction.
_FILLER = re.compile(
    r"\b(please|for me|real quick|if you (can|could|would)|would you|could you|"
    r"can you|i (want|need|would like) (you )?to|go ahead and|just|kinda|sorta|"
    r"like|um+|uh+|er+|hmm+|you know|okay|ok)\b",
    re.I,
)
_PUNCT = re.compile(r"[^\w\s'&/.:-]")
_WS = re.compile(r"\s+")

# Whisper runs two words together when the pause between them is short — "H,
# open Chrome" comes back as "H OpenChrome", and the launch rule needs a space
# to find its verb. Split only where both halves are substantial: that leaves
# "eJPT" and "eCPPT" alone, which a bare camel-case rule would break into "e
# JPT", and it correctly recovers "ZipRecruiter" as two words.
_GLUED = re.compile(r"(?<=[a-z]{3})(?=[A-Z][a-z]{2})")


def normalize(text: str) -> str:
    t = _GLUED.sub(" ", text.strip()).lower()
    for src, dst in _CONTRACTIONS.items():
        t = t.replace(src, dst)
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def strip_filler(text: str) -> str:
    return _WS.sub(" ", _FILLER.sub(" ", text)).strip(" ,.")


# --- wake word -------------------------------------------------------------

_LEADING_JUNK = re.compile(r"^(hey|hi|hello|yo|ok|okay|so|um+|uh+)\s+", re.I)

# Whisper hyphenates a single-letter address onto the word after it: a spoken
# "H, stop" comes back as "H-Stop", which normalisation keeps as one token
# because a hyphen is a legitimate word character in "well-known" and
# "twenty-five". Split it only where the left side is one or two characters,
# which is the shape of a name and never the shape of a compound word.
_LETTER_HYPHEN = re.compile(r"^([a-z0-9]{1,2})[-–]\s*")

# Unambiguous direct address. Whisper drops or mangles a one-syllable name often
# enough that requiring it for these would make HIVE unusable — and nobody says
# "are you there" to the room. Deliberately narrow: presence checks only, no
# commands, so a misfire costs one spoken "Right here." and nothing else.
_DIRECT_ADDRESS = re.compile(
    r"^(are|can|do)?\s*you\s+(there|here|awake|up|on|listening|alive|with me|"
    r"hear me|copy|read me)\b|"
    r"^(are|is)\s+you\s+there\b|"
    r"^you\s+(there|awake|listening)\b", re.I)


def detect_wake(text: str, cfg) -> tuple[bool, str]:
    """Was HIVE addressed, and what is left after removing its name?

    Whisper mishears a one-syllable name constantly — observed in real sessions:
    "Hi", "High", "I've", "5", "Thrive". The strong list fires alone; the weak
    list holds real English words that only count when followed by more speech,
    otherwise a stray "hi" would wake it forever.
    """
    t = _LETTER_HYPHEN.sub(r"\1 ", normalize(text))
    if not t:
        return False, ""

    strong = [w.lower() for w in cfg.get("wake.strong_variants", [])]
    weak = [w.lower() for w in cfg.get("wake.weak_variants", [])]
    threshold = int(cfg.get("wake.fuzzy_threshold", 86))

    def match(words: list[str]) -> tuple[bool, str] | None:
        if not words:
            return None
        first = words[0].strip(".,!?")
        rest = " ".join(words[1:]).strip()
        if first in strong:
            return True, rest
        if first in weak and rest:
            return True, rest
        if len(first) >= 3:
            for cand in strong:
                if fuzz.ratio(first, cand) >= threshold:
                    return True, rest
        # "ok hive, ..." / "so hive ..." — name in second position.
        if len(words) >= 2 and words[1].strip(".,!?") in strong:
            return True, " ".join(words[2:]).strip()
        return None

    # Check the raw text FIRST. Stripping leading junk up front would delete
    # "hi" — which is both filler and one of the most common mishearings of the
    # name — and then report that no wake word was present.
    hit = match(t.split())
    if hit:
        return hit
    hit = match(_LEADING_JUNK.sub("", t).split())
    if hit:
        return hit

    if _DIRECT_ADDRESS.search(t):
        return True, t
    return False, t


# --- barge-in --------------------------------------------------------------
# What counts as "be quiet" when HIVE is *already talking*. Narrower than the
# `stop` intent below, and anchored to the start of the utterance, because this
# one is matched against audio captured while HIVE's own voice is coming out of
# the speakers — so the microphone is hearing both people at once and a loose
# pattern would let HIVE interrupt itself mid-sentence.
#
# Anchoring is what does most of the work: HIVE says "hold on" and "wait" often
# enough, but almost never as the first words of a phrase.
_INTERRUPT = re.compile(
    r"^(stop|stop it|stop stop|shut up|be quiet|quiet|silence|hush|"
    r"enough|that is enough|cancel|nevermind|never mind|forget it|"
    r"hold on|wait|pause|abort|shush|no no)\b", re.I)


def is_interrupt(text: str) -> bool:
    """Is this "stop talking", said over the top of HIVE talking?

    Takes the text with the wake word already removed, so "HIVE stop" arrives
    here as "stop". A bare "stop" counts too — by the time someone is talking
    over an assistant they should not have to address it first — and the
    protection against HIVE triggering this on its own voice is the anchor
    above plus the echo check at the call site, not the wake word.
    """
    return bool(_INTERRUPT.match(strip_filler(normalize(text))))


# --- sites that can be searched by URL -------------------------------------
# Spoken name -> key in `search_sites` in config/apps.yaml, which holds the
# actual query template. The split is the same one the launcher already makes:
# what people say lives in code because the grammar has to match it, and where it
# points lives in configuration because that is what changes.
#
# Kept to names that are unambiguous when spoken. "Target" is a shop and also an
# ordinary word this grammar already uses as a slot name; matching it here would
# turn "find the target of the scan" into a shopping trip.
_SEARCH_SITES = {
    "youtube": "youtube", "you tube": "youtube",
    "best buy": "bestbuy", "bestbuy": "bestbuy",
    "amazon": "amazon",
    "newegg": "newegg", "new egg": "newegg",
    "ebay": "ebay", "e bay": "ebay",
    "walmart": "walmart", "wal mart": "walmart",
    "micro center": "microcenter", "microcenter": "microcenter",
    "google": "google",
    "wikipedia": "wikipedia",
    "reddit": "reddit",
    "github": "github", "git hub": "github",
    "stack overflow": "stackoverflow", "stackoverflow": "stackoverflow",
    "arxiv": "arxiv", "archive x": "arxiv",
    "google scholar": "scholar", "scholar": "scholar",
    "hacker news": "hackernews",
    "exploit db": "exploitdb", "exploitdb": "exploitdb",
}
# Longest first, so "best buy" is matched as itself rather than leaving "buy"
# to be picked up by a shorter alias elsewhere in the alternation.
_SITE_ALTS = "|".join(re.escape(a) for a in
                      sorted(_SEARCH_SITES, key=len, reverse=True))


# --- intent patterns -------------------------------------------------------
# Order matters: first match wins, so specific patterns precede general ones.

_PATTERNS: list[tuple[str, Category, re.Pattern]] = [
    # -- HIVE-Q: the override, so it beats everything ------------------------
    # First in the table, ahead of `stop` and ahead of media transport, because
    # "end query" has to work from inside a running query — which is exactly the
    # state where HIVE is busiest and where every other rule is competing for the
    # same words. "stop the query" must reach the queue, not the music player.
    ("query_end", Category.CONTROL, re.compile(
        r"\b(end|stop|kill|abort|cancel|halt|scrap)\s+(the\s+|this\s+|your\s+)?"
        r"(quer(?:y|ie)s?|queue|q)\b|"
        r"\bquer(?:y|ie)s?\s+(off|stop|end)\b")),
    ("query_run", Category.CONTROL, re.compile(
        r"\b(run|start|begin|go|execute|work through|kick off)\s+"
        r"(the\s+|this\s+|my\s+|your\s+)?(quer(?:y|ie)s?|queue)\b|"
        r"\bwork (?:through|down) (?:the )?(?:quer(?:y|ie)s?|queue|list)\b")),
    ("query_open", Category.CONTROL, re.compile(
        r"\b(open|show|bring up|pull up|give me|launch)\b[^.]*\b"
        r"(hive.?q|the quer(?:y|ie)s?|query window|query pane|task quer(?:y|ie)s?)\b|"
        r"\bhive.?q\b")),

    # -- media transport -----------------------------------------------------
    # Ahead of the control block because "stop the music" and "pause" must reach
    # the player, while a bare "stop" still has to interrupt HIVE mid-sentence.
    ("media_named", Category.MEDIA, re.compile(
        r"\b(play|put on|start)\s+(?P<what>.+?)\s*"
        r"(\bon\s+(spotify|the speakers?)\b|\bplaylist\b|$)")),
    ("media_pause", Category.MEDIA, re.compile(
        r"^(pause|hold)\b(?!\s+on\b)|"
        r"\b(pause|stop|halt)\s+(the\s+)?(music|song|track|playback|spotify|player|audio)\b")),
    ("media_next", Category.MEDIA, re.compile(
        r"\b(next|skip)(\s+(song|track|one))?\b|\bskip (this|it)\b")),
    ("media_prev", Category.MEDIA, re.compile(
        r"\b(previous|last|go back a?)\s*(song|track)\b|\bplay (that|it) again\b")),
    ("media_now", Category.MEDIA, re.compile(
        r"\bwhat('?s| is)? (this |that )?(song|track|playing)\b|"
        r"\bwho('?s| is) this\b|\bwhat am i listening to\b")),
    ("media_play", Category.MEDIA, re.compile(
        r"^(play|resume|unpause)\b|\b(resume|keep playing|start) (the )?(music|playback)\b|"
        r"\bplay (some )?music\b")),

    # -- night mode ----------------------------------------------------------
    ("night_mode", Category.CONTROL, re.compile(
        r"\bnight ?mode\b|\bgo (?:in)?to night\b|\bnight time mode\b")),

    # -- HIVE terminal -------------------------------------------------------
    # Before `launch`, otherwise "open your terminal" resolves to Ptyxis.
    ("terminal_open", Category.CONTROL, re.compile(
        r"\b(open|show|bring up|pull up|give me)\b[^.]*\b(your|the hive|hive)\s+"
        r"(terminal|console|shell|command line)\b|"
        r"\bopen your terminal\b|\bhive terminal\b")),
    ("terminal_close", Category.CONTROL, re.compile(
        r"\b(close|hide|dismiss)\b[^.]*\b(your|the)\s+(terminal|console)\b")),

    # -- control: must beat everything else so "stop" always stops -----------
    ("stop", Category.CONTROL, re.compile(
        r"^(stop|quiet|shut up|be quiet|silence|nevermind|never mind|cancel|"
        r"forget it|hold on|wait)\b")),
    ("sleep", Category.CONTROL, re.compile(
        r"\b(go to sleep|stand down|dismissed|that is all|nothing else|"
        r"we are done|thank you that is all)\b")),
    ("shutdown", Category.CONTROL, re.compile(
        r"\b(shut ?down|power ?off|power down|turn (yourself )?off|close yourself|"
        r"close the app(lication)?|quit|exit|end (the )?session|sign off|"
        r"log off|goodnight|good night|see you later)\b")),

    # -- confirmation --------------------------------------------------------
    ("yes", Category.CONTROL, re.compile(
        r"^(yes|yeah|yep|yup|sure|go|go ahead|do it|confirm|confirmed|affirmative|"
        r"proceed|run it|please do|absolutely|correct|right)\b")),
    ("no", Category.CONTROL, re.compile(
        r"^(no|nope|nah|negative|do not|stop|cancel|skip it|leave it|forget it)\b")),

    # -- the status briefing -------------------------------------------------
    # Ahead of both rules it could otherwise lose to, and the ordering is the
    # whole reason it sits here rather than next to `system_query`:
    #
    #   `presence` matches "status check", so anything reached after it that
    #   mentions a status is competing with "are you there".
    #   `system_query` matches "system status", so "give me a system status
    #   update" would be taken as a request for a disk reading.
    #
    # Narrow on purpose in the other direction: a briefing needs one of the
    # report nouns, so neither of those two rules loses a phrasing to this one.
    # "status check" is still presence and "system status" is still a reading.
    ("status_update", Category.SYSTEM, re.compile(
        r"\bstatus\s+(?:update|report|briefing|brief|rundown|summary)\b|"
        r"\b(?:update|briefing|rundown|debrief)\s+on\s+(?:the\s+)?"
        r"(?:system|everything|things)\b|"
        r"\bsitrep\b|"
        r"\bwhat\s+is\s+(?:your|the)\s+(?:current\s+)?status\b|"
        r"\bbring me up to speed\b|"
        r"\bwhere\s+(?:do\s+)?things\s+stand\b")),

    # -- presence ------------------------------------------------------------
    ("presence", Category.PRESENCE, re.compile(
        r"\b(are you (there|here|awake|up|with me|listening|online|on|alive)|"
        r"you (there|awake|up|with me|listening|online)|"
        r"(can|do) you (hear|copy|read) me|"
        r"do you (copy|read me)|still (there|with me)|status check|check in)\b")),
    ("presence_bare", Category.PRESENCE, re.compile(
        r"^(hello|hey|hi|yo|good morning|good evening|morning|evening)$")),

    # -- voice control -------------------------------------------------------
    ("voice_change", Category.CONTROL, re.compile(
        r"\b(switch|change|use|try|swap)\b[^.]*\bvoice\b|"
        r"\b(sound|speak|talk)\b.*\b(american|british|english|softer|lower|different)\b")),
    ("voice_list", Category.CONTROL, re.compile(
        r"\bwhat voices\b|\blist (the )?voices\b|\bwhich voices\b")),

    # -- time ----------------------------------------------------------------
    ("time", Category.TIME, re.compile(
        r"\b(what is the (time|date|day)|what time is it|what day is it|"
        r"today is|what is today)\b")),

    # -- compound: open something AND put a question into it -----------------
    # A dependent two-step: step 2 needs step 1's target. Handled as one intent
    # rather than by generic decomposition, because splitting it would produce a
    # second step with nothing to act on.
    ("launch_ask", Category.LAUNCH, re.compile(
        r"\b(?:open|launch|start|pull up|fire up|bring up)\b\s+(?P<target>.+?)\s+"
        r"(?:and\s+)?(?:ask(?:ing)?(?:\s+it)?|with the question|and say|prompt(?:ing)?"
        r"|and ask(?:ing)?)\s+(?P<question>.+)$")),

    # -- job search ----------------------------------------------------------
    # Naming a board is itself the strongest signal. An earlier version required
    # the word "job" or "listing" and missed the operator's own example —
    # "check indeed and ziprecruiter for overnight help desk" never says either.
    ("job_search", Category.RESEARCH, re.compile(
        r"\b(check|search|look|find|show me|pull up|open|any|browse)\b[^.]*\b"
        r"(indeed|ziprecruiter|zip recruiter|glassdoor|linked ?in)\b|"
        r"\b(indeed|ziprecruiter|zip recruiter|glassdoor)\b[^.]*\b(for|jobs?|listings?)\b|"
        r"\b(check|search|look for|find|show me|pull up|any)\b[^.]*\b"
        r"(job|jobs|listing|listings|openings?|positions?|roles?|postings?)\b")),

    # -- granted workspaces --------------------------------------------------
    # Folder names are resolved against what is actually granted rather than
    # parsed out of the transcript, so a mangled name fails to match instead of
    # matching the wrong directory.
    ("workspace_organize", Category.WORKSPACE, re.compile(
        r"\b(organi[sz]e|arrange|restructure|mirror|lay ?out|sort)\b.*"
        r"\b(same|like|match|way)\b|"
        r"\b(organi[sz]e|arrange|restructure)\b.*\b(folder|directory|contents)\b")),
    ("workspace_list", Category.WORKSPACE, re.compile(
        r"\bwhat (folders?|directories|access)\b.*\b(have|got|granted|see)\b|"
        r"\bwhich folders?\b|\bwhat do you have access to\b|"
        r"\bwhat can you (see|access)\b")),
    ("workspace_describe", Category.WORKSPACE, re.compile(
        r"\b(what('?s| is) in|describe|tell me about|look at|inspect)\b.*"
        r"\b(folder|directory)\b")),

    # -- the network ---------------------------------------------------------
    # Ahead of `research_ask` on purpose. "research options for me on Best Buy
    # for keyboards" contains the word *research* and is not a request for
    # literature — naming a shop is the stronger signal, and the site rules below
    # all require a site to be named, so they cannot capture a general question.
    #
    # Ahead of `launch` for the same reason `research_ask` is: "open YouTube and
    # type X into the search bar" is one request with a payload, and the launch
    # rule would hand the entire tail to the application resolver.
    ("site_search", Category.RESEARCH, re.compile(
        rf"\b(?:{_SITE_ALTS})\b[^.]*?\b(search|searching|type|typing|look up|"
        rf"find|for|research|options|compare|price|prices|shop|browse)\b|"
        rf"\b(search|searching|type|typing|look up|find|research|options|compare|"
        rf"price|prices|shop|browse)\b[^.]*?\b(?:{_SITE_ALTS})\b")),
    # An explicit request to go and look, as opposed to a question HIVE could
    # answer from what it knows. The lookahead keeps "look up my notes" with the
    # vault, which is a different verb pointed at a different place.
    ("web_search", Category.RESEARCH, re.compile(
        r"\b(?:search|look)\s+(?:it\s+|that\s+|this\s+)?(?:up\s+)?(?:on\s+)?"
        r"(?:the\s+)?(?:web|internet|online)\b|"
        r"\b(?:search|look\s+up|check|find)\b(?!\s+(?:my|the)\s+"
        r"(?:notes?|vault|memory))[^.]*\b(?:online|on the web|on the internet)\b|"
        r"\bwhat(?: is|'?s|\s+are)\s+(?:the\s+)?"
        r"(?:latest|newest|current|recent)\b[^.]*\b(?:news|on|about|with|for)\b|"
        r"\bgoogle\s+(?:me\s+)?(?!docs?\b|sheets?\b|slides?\b|drive\b|"
        r"calendar\b|chrome\b)\w")),

    # -- vault write: a document, not a capture ------------------------------
    # Ahead of `note_add`, which is the quick-capture path — "note that the
    # exam is Friday" writes one bullet and is done. This is the other request
    # entirely: take everything that has been discussed and write it up. It used
    # to match nothing at all, so it fell through to conversation, and the model
    # said it was creating the file and then said it again, because saying so
    # was the only thing that ever happened. Nothing was ever written.
    #
    # Also ahead of `research_ask`, which is the more general rule: "write up
    # the research we did" names a document *and* mentions research, and the
    # instruction to write the file is the more specific reading of the two.
    #
    # Required: a document word, or a create verb pointed at the vault, or a
    # create verb with a backward reference ("everything we talked about"). A
    # bare "make a note" is deliberately NOT enough — that is a capture.
    ("note_write", Category.VAULT_WRITE, re.compile(
        # "create a markdown file", "write an md note", "make a document"
        r"\b(?:make|create|write|draft|generate|build|put together|compile|"
        r"assemble|start|save|log|document|type|summari[sz]e|recap|turn|"
        r"convert|export|dump)\b"
        r"(?:\s+\w+){0,4}?\s+"
        r"(?:markdown|md|document|file|report|write ?up|summary|recap|page|brief)\b|"
        # The phrasal verb, with the particle on either side of its object and
        # the object allowed to run a few words: "write up the research we did",
        # "write this up", "write this whole thing up", "type it down". On its
        # own that is already an unambiguous request for a written record — the
        # lookahead only removes the things it can mean that are not a vault
        # document.
        r"\b(?:write|type|jot)\s+"
        r"(?!(?:\S+\s+){0,3}?(?:e-?mails?|messages?|repl(?:y|ies)|letters?|"
        r"texts?|posts?|comments?|dms?)\b)"
        r"(?:\S+\s+){0,4}?(?:up|down)\b|"
        r"\bdocument\s+(?:this|that|everything|all of (?:this|that)|the session|"
        r"our|the conversation)\b|"
        # anything aimed explicitly at Obsidian or the vault
        r"\b(?:in ?to|in|on)\s+(?:obsidian|my vault|the vault|my notes)\b|"
        r"\bobsidian\s+(?:note|file|page|document)\b|"
        # a create verb with a backward reference to the session
        r"\b(?:make|create|write|draft|generate|save|log|compile)\b[^.]{0,60}?"
        r"\b(?:everything|all of (?:it|this|that)|what we (?:just )?"
        r"(?:talked about|discussed|covered|went over|said)|"
        r"this (?:whole )?(?:conversation|session|discussion)|"
        r"our (?:conversation|discussion))\b")),

    # -- research, and the launch verbs it shares ----------------------------
    # Ahead of `launch`, and this position is the whole point of the pattern.
    #
    # "pull up some university articles on human-sounding voice agents" is a
    # research question that happens to open with a launch verb. The launch rule
    # matched it, handed the entire rest of the sentence to the resolver as an
    # application name, and the fuzzy scan over installed desktop entries scored
    # "Passwords and Keys" at 85 against it — so HIVE answered "got it, opening
    # passwords and keys" and opened Seahorse. Verified by replaying the exact
    # utterance through the resolver.
    #
    # Naming a kind of *reading* is decisive: nothing on this machine is called
    # an article or a paper, so a request that names one is asking about a
    # subject rather than for a program.
    #
    # This used to resolve to CONVERSE — a pure veto on the launch rule below it,
    # because with no network there was nothing else it could honestly become.
    # Now that there is one it is a real intent, and the objective hemisphere
    # decides which: online it searches, offline it falls straight back through
    # to conversation, which is byte-for-byte the old behaviour. The property the
    # rule was written for is unchanged either way — it must never reach `launch`.
    ("research", Category.RESEARCH, re.compile(
        r"\b(articles?|papers?|stud(?:y|ies)|research|journals?|literature|"
        r"publications?|white ?papers?|preprints?|citations?|references?|"
        r"documentation|peer[- ]reviewed|scholarly|academic)\b")),

    # -- vault write ---------------------------------------------------------
    ("note_add", Category.VAULT_WRITE, re.compile(
        r"\b((make|take|write|add|create|jot|start)\s+(me\s+)?(a\s+)?note|"
        r"note (that|this|down)|"
        r"(remember|save|log|record|capture) (that|this)|"
        # "add <anything> to my todo" — the payload sits between verb and target,
        # so this has to admit arbitrary content, not just "that"/"this".
        r"(add|put|stick)\s+.{0,90}?\s*(?:to|on|onto|in) (my )?"
        r"(todo|to do|to-do|list|tasks?|notes?))\b")),

    # -- vault read ----------------------------------------------------------
    ("note_recall", Category.VAULT_READ, re.compile(
        r"\b(what did (we|i) (say|talk about|discuss|decide)|"
        r"do you remember|remind me (about|what|of)|"
        r"(look up|search|find|pull up|check) (my |the )?(notes|vault|memory)|"
        r"what do you (know|have) about)\b")),

    # -- system --------------------------------------------------------------
    ("system_query", Category.SYSTEM, re.compile(
        r"\b(how much (disk|space|storage|memory|ram)|"
        r"(disk|storage) (space|usage|free)|"
        r"(cpu|gpu|memory|ram|vram) (usage|load|temp|temperature)|"
        r"(what is|check) (my |the )?(battery|uptime|temperature|load)|"
        r"how (hot|warm) is|system (status|info|stats))\b")),

    # -- launch: most general, so it goes last --------------------------------
    ("launch", Category.LAUNCH, re.compile(
        r"\b(open( up)?|launch|start( up)?|fire up|bring up|pull up|boot( up)?|"
        r"load up|show me|get me|take me to|go to|navigate to|new)\b\s+(?P<target>.+)")),
]


def classify(cleaned: str) -> Intent:
    """Match the post-wake-word text against the grammar."""
    t = strip_filler(normalize(cleaned))
    if not t:
        # Bare "HIVE", with nothing after it — the name said on its own while the
        # operator works out what they want. Kept distinct from `presence`, which
        # is an actual question ("are you there?") and deserves an actual answer.
        #
        # This one usually does not: the visualizer brightens and the header
        # reads LISTENING the moment it wakes, so a spoken "Still here" is HIVE
        # telling you something the screen already told you, over the top of you
        # starting your sentence. See wake.acknowledge in config/hive.toml.
        return Intent("wake_only", Category.PRESENCE, {}, 1.0, cleaned, t)

    for name, cat, pat in _PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        slots: dict = {}
        if name == "research":
            slots = {"topic": parse_research_topic(t)}
            if not slots["topic"]:
                # Nothing to search for — the word "research" used as a noun in
                # an ordinary question ("is that research solid?"). Hand it to
                # conversation, which is what it was before there was a network,
                # and keep the veto on `launch` that this rule exists for.
                return Intent("converse", Category.RESEARCH, {}, 0.0, cleaned, t)
        elif name == "site_search":
            slots = parse_site_search(t, cleaned)
            if not slots.get("site") or not slots.get("query"):
                continue        # a site was named in passing, not searched
        elif name == "web_search":
            slots = {"query": parse_web_query(t)}
            if not slots["query"]:
                continue
        elif name == "launch":
            slots = parse_launch_target(m.group("target"), t)
            if not slots.get("target"):
                continue        # "open" with nothing after it is not a launch
        elif name == "note_write":
            slots = parse_document_request(t)
        elif name == "note_add":
            slots = {"content": _note_content(t, m)}
        elif name == "note_recall":
            slots = {"query": _after(t, m) or t}
        elif name == "night_mode":
            if re.search(r"\b(off|disable|stop|exit|leave|cancel|end)\b", t):
                slots = {"want": False}
            elif re.search(r"\b(auto|schedule|normal|default)\b", t):
                slots = {"want": None}
            else:
                slots = {"want": True}
        elif name == "launch_ask":
            target = parse_launch_target(m.group("target"), t)
            question = m.group("question").strip(" ,.").strip("\"'“”‘’")
            if not target.get("target") or len(question) < 4:
                continue
            slots = dict(target)
            slots["question"] = question
        elif name == "job_search":
            slots = parse_job_query(t)
        elif name == "media_named":
            what = (m.groupdict().get("what") or "").strip(" ,.")
            what = _ARTICLES.sub("", what).strip()
            # "play music" / "play something" is a generic resume, not a request
            # for a specific title — route it to plain playback instead.
            if not what or re.fullmatch(
                    r"(some )?(music|something|a song|songs|tunes|audio)", what):
                return Intent("media_play", Category.MEDIA, {}, 1.0, cleaned, t)
            slots = {"what": what}
        elif name == "voice_change":
            slots = {"request": t}
        # Deterministic match: full confidence by construction.
        return Intent(name, cat, slots, 1.0, cleaned, t)

    return Intent("converse", _converse_category(t), {}, 0.0, cleaned, t)


def _after(text: str, m: re.Match) -> str:
    """Whatever follows the matched trigger — the payload of a note command."""
    tail = text[m.end():].strip(" ,.:;-")
    tail = re.sub(r"^(that|this|down|about|saying|which is)\b\s*", "", tail).strip()
    return tail


# "add <content> to my todo" puts the payload BEFORE the trigger, unlike every
# other note form where it trails. Handled separately rather than by widening
# _after, which would silently mangle the common case.
_INFIX_NOTE = re.compile(
    r"\b(?:add|put|stick)\s+(?P<content>.+?)\s+(?:to|on|in)\s+(?:my\s+)?"
    r"(?:todo|to do|to-do|list|tasks?|notes?)\b", re.I)

# Leftovers that mean "no payload was given" rather than a real note body.
_EMPTY_PAYLOAD = re.compile(r"^(list|it|that|this|todo|to do|to-do|tasks?|notes?)?$", re.I)


def _note_content(text: str, m: re.Match) -> str:
    infix = _INFIX_NOTE.search(text)
    if infix:
        content = infix.group("content").strip(" ,.")
        # "add that to my todo" — "that" refers to prior context, not a note body.
        if not _EMPTY_PAYLOAD.match(content):
            return content
        return ""
    tail = _after(text, m)
    return "" if _EMPTY_PAYLOAD.match(tail.strip()) else tail


def _converse_category(t: str) -> Category:
    """Pick a status word family for free-form questions."""
    if re.search(r"\b(write|draft|compose|word|rewrite|edit|polish)\b", t):
        return Category("writing")
    if re.search(r"\b(why|how does|explain|analyz|analys|debug|review|break down)\b", t):
        return Category("analysis")
    if re.search(r"\b(compare|versus|vs|difference|pros and cons|trade ?off)\b", t):
        return Category("synthesis")
    if re.search(r"\b(look up|search|find|research|who is|what is)\b", t):
        return Category("research")
    return Category.CONVERSE


# --- launch slot parsing ---------------------------------------------------

_TRAILING = re.compile(
    r"\s+\b(window|tab|app|application|program|browser|page|thing|one)\b\s*$", re.I)
_ON_BROWSER = re.compile(r"\s+\b(?:on|in|using|with|through)\s+(?P<b>[\w\s]+?)\s*$", re.I)
_ARTICLES = re.compile(r"^\s*(a|an|the|my|some|another|new)\s+", re.I)

_BROWSER_WORDS = {
    "chrome": "chrome", "google chrome": "chrome", "the browser": "chrome",
    "browser": "chrome", "firefox": "firefox", "mozilla": "firefox",
    "fire fox": "firefox",
}

# An application is a name. It is not a sentence, and it is never a description
# of a subject. These are the shapes that prove the "target" is neither.
#
# The guard exists because launch is the last and most permissive rule in the
# grammar, so every unrecognised imperative lands on it with the whole utterance
# as its argument — and the resolver's job is to find *something*, which on a
# long enough string it always does. Failing to parse a target sends the turn to
# conversation, which is the correct home for anything this rejects.
MAX_TARGET_WORDS = 5
MAX_TARGET_CHARS = 44
_NOT_A_TARGET = re.compile(
    r"\b(about|regarding|concerning|covering|explaining|related to|involving|"
    r"that (?:are|is|show|shows|talk|talks|explain|explains|cover|covers)|"
    r"which|because|so that|whether|how to|why|versus|vs)\b|"
    # A trailing prepositional phrase is a subject, not a name: "articles ON
    # voice agents", "news ON the exam". Browsers were already stripped above,
    # so an "on" surviving to here is never "on chrome".
    r"\s\b(?:on|about|for)\s+\w+\s+\w+", re.I)


# --- job search slot parsing ----------------------------------------------

_SITES = {
    "indeed": "indeed", "ziprecruiter": "ziprecruiter", "zip recruiter": "ziprecruiter",
    "linkedin": "linkedin", "linked in": "linkedin", "glassdoor": "glassdoor",
}
_RADIUS = re.compile(r"\b(?:within\s+)?(\d{1,3})\s*(?:mi|mile|miles)\b", re.I)
_DAYS = re.compile(
    r"\b(?:past|last|within the last)\s+(\d{1,2})?\s*(day|days|week|weeks|month)\b", re.I)
# Ordered most-specific first. A single permissive "in|near (.+)" pattern reads
# "in the past week within 25 miles of San Diego" as a location called
# "past week within 25 miles of san diego" — the radius and recency phrases sit
# between the preposition and the actual place name.
_LOCATION_FORMS = [
    re.compile(r"\bwithin\s+\d{1,3}\s*(?:mi|miles?)\s+of\s+(?:the\s+)?"
               r"(?P<loc>[A-Za-z][A-Za-z\s]{1,38}?)(?:\s+area)?\s*(?:$|[,.]|\bfor\b)", re.I),
    # A place name ends where the next qualifier begins, so "in|within|posted"
    # are terminators as well as "for"/"that" — otherwise "near los angeles in
    # the last 3 days" captures the whole tail and gets rejected as a non-place.
    re.compile(r"\b(?:in|near|around)\s+(?:the\s+)?"
               r"(?P<loc>[A-Za-z][A-Za-z\s]{1,38}?)(?:\s+area)?\s*"
               r"(?:$|[,.]|\bfor\b|\bthat\b|\bin\b|\bwithin\b|\bposted\b|\blisted\b)",
               re.I),
]
# Words that prove a candidate is a time or distance phrase, not a place.
_NOT_A_PLACE = re.compile(
    r"\b(past|last|week|weeks|day|days|month|hour|mile|miles|radius|within|"
    r"posted|listed|recent|new|up)\b", re.I)
_ROLE = re.compile(r"['\"“]([^'\"”]{3,60})['\"”]")


def parse_job_query(t: str) -> dict:
    """Pull role, location, radius and recency out of a spoken job request."""
    slots: dict = {"sites": [], "role": "", "location": "", "radius": 25, "days": 7}

    for name, key in _SITES.items():
        if name in t and key not in slots["sites"]:
            slots["sites"].append(key)

    m = _ROLE.search(t)
    if m:
        slots["role"] = m.group(1).strip()
    else:
        for pattern in (
            r"\bfor\s+(?P<r>[\w\s+/-]{3,45}?)\s*(?:\bthat\b|\bin\b|\bnear\b|\bwithin\b|[,.]|$)",
            # "any it support roles near ..." / "check help desk jobs in ..."
            r"\b(?:any|some)?\s*(?P<r>[\w\s+/-]{3,45}?)\s+"
            r"(?:jobs?|roles?|positions?|listings?|openings?|postings?)\b",
        ):
            m = re.search(pattern, t)
            if m:
                slots["role"] = m.group("r").strip()
                break

    # The role must not absorb the location or radius. "junior analyst within 50
    # miles of riverside" is a role of "junior analyst"; everything from the
    # preposition onward belongs to other slots.
    if slots["role"]:
        role = re.split(r"\s+\b(?:within|near|around|in)\b\s+", slots["role"])[0]
        role = re.sub(r"\s+\b(?:jobs?|roles?|positions?|listings?|openings?)\b\s*$",
                      "", role, flags=re.I)
        role = re.sub(r"^\b(?:any|some|the|a)\b\s+", "", role, flags=re.I).strip()
        # The greedy role pattern can start at a command verb ("show me overnight
        # positions" -> "show me overnight"). Strip those from the front.
        role = re.sub(r"^\b(?:show me|show|check|search|find|look for|pull up|"
                      r"open|browse|get me|any)\b\s+", "", role, flags=re.I).strip()
        slots["role"] = role

    m = _RADIUS.search(t)
    if m:
        slots["radius"] = int(m.group(1))

    m = _DAYS.search(t)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2).lower()
        slots["days"] = n * (7 if unit.startswith("week") else 30 if unit == "month" else 1)

    for pattern in _LOCATION_FORMS:
        for m in pattern.finditer(t):
            loc = m.group("loc").strip().rstrip(".")
            if _NOT_A_PLACE.search(loc) or not (2 < len(loc) < 40):
                continue          # a time or distance phrase, keep looking
            slots["location"] = loc
            break
        if slots["location"]:
            break
    return slots


def parse_launch_target(target: str, full: str) -> dict:
    """Pull the real target out of a spoken launch phrase.

    "open up a new google docs window for me on google chrome"
      -> target='google docs', browser='chrome', want_window=True
    """
    t = strip_filler(target).strip(" ,.")
    slots: dict = {"want_window": False, "browser": None}

    m = _ON_BROWSER.search(t)
    if m:
        cand = m.group("b").strip().lower()
        if cand in _BROWSER_WORDS:
            slots["browser"] = _BROWSER_WORDS[cand]
            t = t[: m.start()].strip()

    if re.search(r"\bnew\b", full) or re.search(r"\bwindow\b", t):
        slots["want_window"] = True

    # Strip trailing nouns repeatedly: "docs window app" -> "docs"
    prev = None
    while prev != t:
        prev = t
        t = _TRAILING.sub("", t).strip()

    # Strip leading articles repeatedly: "a new the doc" -> "doc"
    prev = None
    while prev != t:
        prev = t
        t = _ARTICLES.sub("", t).strip()

    t = t.strip(" ,.")
    # Refuse rather than guess. An empty target drops the launch rule and the
    # utterance carries on to conversation, which is where a sentence belongs.
    if (len(t.split()) > MAX_TARGET_WORDS or len(t) > MAX_TARGET_CHARS
            or _NOT_A_TARGET.search(t)):
        slots["target"] = ""
        slots["rejected"] = t
        return slots

    slots["target"] = t
    return slots


# --- document request slot parsing -----------------------------------------

# What the write-up should be *about*. Absent, it covers the session, which is
# the common case: "write up everything we talked about" names no subject
# because the subject is the conversation.
_DOC_SUBJECT = re.compile(
    r"\b(?:about|on|covering|regarding|of|for|detailing|documenting)\s+"
    r"(?P<subject>[\w][\w\s'&+/-]{2,70}?)\s*"
    r"(?:$|[,.]|\bin ?to\b|\bin\b|\bwith\b|\band (?:put|save|write)\b)", re.I)

# Phrases that mean "the whole conversation", not a subject.
_WHOLE_SESSION = re.compile(
    r"\b(everything|all of (?:it|this|that)|the (?:whole )?(?:conversation|"
    r"session|discussion|thing)|what we (?:just )?(?:talked about|discussed|"
    r"covered|went over|said)|our (?:conversation|discussion))\b", re.I)

# An explicit request for depth. Jay's standing complaint about the vault is
# that the notes are ten lines long, so "detailed" is the default rather than
# the exception — but saying it should still be recorded, because it is also
# the switch that turns off any future trimming.
_EXHAUSTIVE = re.compile(
    r"\b(detail(?:ed)?|in depth|in-depth|thorough(?:ly)?|complete(?:ly)?|full|"
    r"everything|comprehensive|exhaustive|long|verbose|all of it)\b", re.I)


# --- network slot parsing ---------------------------------------------------

# What a research request is *about*. First preposition wins and it captures to
# the end of the string: "articles on Kerberos for the exam" is a search for
# "Kerberos for the exam", which is a better query than either half alone.
_RESEARCH_TOPIC = re.compile(
    r"\b(?:on|about|regarding|concerning|covering|into|around|for)\s+"
    r"(?P<topic>[\w][\w\s'&+/-]{2,90}?)\s*(?:$|[,.?!])")

# Requests that name a kind of reading but no subject. There is nothing to search
# for, so they belong in conversation.
_NO_TOPIC = re.compile(
    r"^(?:it|this|that|them|those|these|me|us|my|the|a|an|any|some|more)$", re.I)


def parse_research_topic(t: str) -> str:
    m = _RESEARCH_TOPIC.search(t)
    if not m:
        return ""
    topic = _WS.sub(" ", m.group("topic")).strip(" ,.")
    topic = _ARTICLES.sub("", topic).strip()
    # Strip the reading-noun if the preposition swept it up ("on the subject of
    # papers about X" is rare, but "for research on X" is not).
    topic = re.sub(r"^(?:research|articles?|papers?|studies|literature)\s+"
                   r"(?:on|about)\s+", "", topic, flags=re.I).strip()
    if len(topic) < 3 or _NO_TOPIC.match(topic):
        return ""
    return topic


# Ordered most-specific first. Each returns the payload to search for; the site
# is resolved separately, because it can appear on either side of the payload
# and in several of these forms it appears in both halves of the sentence.
_SITE_FORMS = [
    # "...type into the search bar 'betterminecraft+ hardcore gameplay'"
    re.compile(r"\bsearch\s*(?:bar|box|field|line)\b\s*[:,\-]?\s*(?P<q>.+)$", re.I),
    # "...type 'x' into the search bar"
    re.compile(r"\b(?:type|put|enter|paste)\s+(?P<q>.+?)\s+in(?:to)?\s+"
               r"(?:the\s+)?search\b", re.I),
    # "search youtube for x" / "check best buy for keyboards"
    re.compile(rf"\b(?:{_SITE_ALTS})\s+(?:search\s+)?for\s+(?P<q>.+)$", re.I),
    # "research options for me on best buy for keyboards similar to ..."
    re.compile(rf"\bon\s+(?:{_SITE_ALTS})\b\s+for\s+(?P<q>.+)$", re.I),
    # "open youtube and search for x" / "go to github and find x"
    re.compile(rf"\b(?:{_SITE_ALTS})\b[^.]*?\b(?:search|look\s+up|find|browse|"
               rf"type|shop)\s+(?:for\s+|up\s+)?(?P<q>.+)$", re.I),
    # "search for x on youtube" / "look up x on github"
    re.compile(rf"\b(?:search|look\s+up|find|check|shop)\s+(?:for\s+)?"
               rf"(?P<q>.+?)\s+(?:on|at|in|through)\s+(?:{_SITE_ALTS})\b", re.I),
]

_QUOTED = re.compile(r"['\"“”‘’]([^'\"“”‘’]{2,120})['\"“”‘’]")
# Payload leftovers that mean the request named a site and nothing to look for.
_EMPTY_QUERY = re.compile(
    r"^(?:it|this|that|them|something|anything|stuff|me|my|the|a|an|for|up|"
    r"bar|box|search|options?)$", re.I)


def _site_in(text: str) -> str:
    """Longest site alias present in the text, as its config key."""
    low = (text or "").lower()
    best = ""
    for alias, key in _SEARCH_SITES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", low) and len(alias) > len(best):
            best = alias
    return _SEARCH_SITES.get(best, "")


def parse_site_search(t: str, raw: str = "") -> dict:
    """Pull the site and the thing to search for out of a spoken or typed request.

    Run against the *raw* text first, and that is not a nicety. `normalize()`
    strips every character outside `\\w\\s'&/.:-`, which quietly deletes the plus
    in "betterminecraft+ hardcore gameplay" — a literal search string typed by
    hand into HIVE-Q, where the whole point is that it goes across exactly as
    written. The normalised text is kept as the fallback for the case the raw
    form does not parse, which is what happens when the request arrived through
    the recogniser and the punctuation was never there to begin with.
    """
    slots: dict = {"site": "", "query": "", "read": False}
    source = (raw or t).strip()
    slots["site"] = _site_in(source) or _site_in(t)
    if not slots["site"]:
        return slots

    query = ""
    # A quoted string is unambiguous: it is the payload, whatever the sentence
    # around it is doing. Checked before the positional forms for that reason.
    q = _QUOTED.search(source)
    if q:
        query = q.group(1).strip()
    else:
        for form in _SITE_FORMS:
            m = form.search(source) or (form.search(t) if raw else None)
            if m:
                query = m.group("q").strip()
                break

    query = query.strip(" ,.;:!?").strip("'\"“”‘’").strip()
    query = re.sub(r"^(?:for|up|the|a|an|me|some)\s+", "", query, flags=re.I).strip()
    # Trailing instruction fragments the greedy tail picks up.
    query = re.sub(r"\s+(?:in|into)\s+(?:the\s+)?search\s*(?:bar|box|field)?\.?$",
                   "", query, flags=re.I).strip()
    if not query or _EMPTY_QUERY.match(query) or len(query) < 2:
        return {"site": slots["site"], "query": "", "read": False}

    # Whether HIVE should read the results back or simply open the page. Naming
    # a comparison or a price is a request for an answer; "open YouTube and
    # search for X" is a request for a page, and reading six video titles aloud
    # to someone who asked for a browser tab is noise.
    slots["read"] = bool(re.search(
        r"\b(option|options|compare|comparison|price|prices|cost|cheapest|"
        r"best|similar|like|alternatives?|recommend|which|review|reviews|"
        r"research|read|tell me|what)\b", source, re.I))
    slots["query"] = query[:180]
    return slots


_WEB_QUERY_FORMS = [
    re.compile(r"\b(?:search|look)\s+(?:the\s+)?(?:web|internet|online)\s+"
               r"(?:for\s+)?(?P<q>.+)$", re.I),
    re.compile(r"\b(?:search|look\s+up|find|check)\s+(?P<q>.+?)\s+"
               r"(?:online|on the web|on the internet)\b", re.I),
    re.compile(r"\bwhat(?:\s+is|'?s|\s+are)\s+the\s+"
               r"(?:latest|newest|current|recent)\b[^.]*?"
               r"\b(?:on|about|with|for)\s+(?P<q>.+)$", re.I),
    re.compile(r"\bgoogle\s+(?:me\s+)?(?P<q>.+)$", re.I),
    re.compile(r"\b(?:search|look\s+up|find)\s+(?:for\s+)?(?P<q>.+)$", re.I),
]


def parse_web_query(t: str) -> str:
    for form in _WEB_QUERY_FORMS:
        m = form.search(t)
        if not m:
            continue
        q = _WS.sub(" ", m.group("q")).strip(" ,.;:!?")
        q = re.sub(r"\s+(?:online|on the web|on the internet)\s*$", "", q, flags=re.I)
        q = _ARTICLES.sub("", q).strip()
        if q and not _EMPTY_QUERY.match(q) and len(q) > 2:
            return q[:180]
    return ""


def parse_document_request(t: str) -> dict:
    """Pull the subject and scope out of a spoken "write this up" request."""
    slots: dict = {"subject": "", "scope": "session", "exhaustive": True}

    if not _WHOLE_SESSION.search(t):
        m = _DOC_SUBJECT.search(t)
        if m:
            subject = _WS.sub(" ", m.group("subject")).strip(" ,.")
            subject = _ARTICLES.sub("", subject).strip()
            # "make a markdown file of it" — a pronoun is not a subject, and
            # "in obsidian" is a destination that the preposition swept up.
            if (2 < len(subject) < 72
                    and not re.fullmatch(
                        r"(it|this|that|them|those|these|us|me|obsidian|"
                        r"(my|the) (vault|notes)|today|now)", subject, re.I)):
                slots["subject"] = subject
                slots["scope"] = "subject"

    slots["exhaustive"] = bool(_EXHAUSTIVE.search(t)) or slots["scope"] == "session"
    return slots
