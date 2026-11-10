"""The status briefing — a written message, not a computed one.

Jay asked for a specific paragraph in reply to "give me a current status
update", and the wording is his. That fixes what this module is and is not: the
state, the two counts and the two directory paths are *authored values* held in
configuration, not readings taken off a running system. There is no incident
tracker behind them and nothing here goes looking for one. A briefing that
counted real failures would be a different feature with a different name, and
inventing one behind this wording would make HIVE assert things it cannot check
— which is the one thing the objective hemisphere exists to prevent.

What this module does own is the gap between the message as written and what
Kokoro can actually say, because those are not the same string. Every claim
below was measured on this machine, by phonemising the candidates through the
same espeak build the voice uses:

  "/hive/database/livefeed/"  espeak reads a slash as the word "slash", so the
                              sentence comes out "check the live feed slash hive
                              slash database slash livefeed slash for further
                              details". A path is something to read, not
                              something to hear. They go to the terminal.

  "Wednesday, Sept. 23rd"     broken twice over. espeak phonemises "Sept." as
                              /sˈɛpt/ — the word "sept", rhyming with "kept",
                              not the month. And the full stop inside the
                              abbreviation reads as the end of a sentence to
                              HIVE's own unit splitter, so the date is delivered
                              as "on Wednesday, September." [150ms, falling]
                              "Twenty-third at 6:41 PM." The month is therefore
                              written out in full here. prosody._ABBREV covers
                              the case where the abbreviation arrives from
                              somewhere else.

  "Stable (recovered)"        a line with no terminal punctuation followed by a
                              blank line. The blank line is whitespace and does
                              not survive despatialise(), and with nothing in
                              its place the state runs straight on into the next
                              sentence as one twenty-word breath. The template
                              closes the sentence itself.

  "low-severity"              phonemised /lˈoʊsəvˈɛɹᵻɾi/, glued into one word
                              under a single stress. Spaced, it is /lˈoʊ
                              səvˈɛɹᵻɾi/, which is how it is said. Hyphens are
                              for the screen.

So a briefing renders twice — `spoken` for the voice and `paths` for the
terminal. That split is not new: the web layer already sends the links HIVE
deliberately does not read aloud to the same place.

Everything here is a deterministic function of the clock and the configuration.
That is load bearing rather than tidy: the audio cache is keyed on the finished
string, so a greeting drawn from a shuffle bag would miss the cache on every
single briefing and pay full synthesis for a line that could have been free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Fallbacks, so a briefing still renders if `status:` is edited out of
# persona.yaml or [status] out of hive.toml. Duplicated wording, deliberately:
# the alternative is a briefing that goes silent on a YAML mistake, and a
# briefing is the one answer where silence is indistinguishable from "nothing is
# wrong". They are not kept in sync automatically — the configured text is what
# is normally spoken, and these only surface when it is missing.
_DEFAULT_STATE = "Stable (recovered)"
_DEFAULT_FEEDS = ("Check the live feed and error log directory "
                  "for further details.")
_DEFAULT_CLEAR = "Nothing has failed since then, and there is nothing waiting on you."
_DEFAULT_CRITICAL = {
    "critical_one": ("there has been 1 critical system failure that requires "
                     "your immediate attention"),
    "critical_many": ("there have been {n} critical system failures that require "
                      "your immediate attention"),
}
_DEFAULT_LOW = {
    "low_one": ("there is {also}1 low severity bug I'd recommend reviewing when "
                "you get a chance"),
    "low_many": ("there are {also}{n} low severity bugs I'd recommend reviewing "
                 "when you get a chance"),
}
_DEFAULT_MESSAGE = {
    "message": ("{greeting}, {operator}. Your last status update was on "
                "{last_update}. Current system state: {state}. {incidents} {feeds}"),
    "message_first": ("{greeting}, {operator}. Current system state: {state}. "
                      "{incidents} {feeds}"),
}


def greeting(now: datetime | None = None) -> str:
    """Time of day, on the boundaries people actually use.

    Deliberately not cfg.timeslot(): that splits the day for persona flavour —
    late_night, early_morning, work_hours, evening — and lumps 9am and 5pm
    together, which is exactly the distinction a greeting turns on. Small hours
    stay "evening" because at 1am that is what a person says; "good morning" at
    1am reads as a clock error, and this message opens with it.
    """
    h = (now or datetime.now()).hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 18:
        return "Good afternoon"
    return "Good evening"


def _ordinal(day: int) -> str:
    """23 -> "23rd". Spoken aloud by espeak as /twˈɛnti θˈɜːd/, verified."""
    if 11 <= day % 100 <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def spoken_datetime(dt: datetime) -> str:
    """"Wednesday, September 23rd at 6:41 PM" — the month never abbreviated.

    The ordinal is substituted before strftime rather than after, so a literal
    "%d" can never survive into the output. The trailing replace strips the
    zero off a padded hour, which is the same idiom base_facts() uses on the
    clock it hands the model.
    """
    return dt.strftime(f"%A, %B {_ordinal(dt.day)} at %I:%M %p").replace(" 0", " ")


def _sentence(text: str) -> str:
    """Capitalise an opening and close it, for a clause assembled from fragments.

    The clause templates are written lower case and unpunctuated because which of
    them opens the paragraph depends on which count is zero, and a fragment that
    arrives mid-paragraph must not carry a capital it did not earn. Closing the
    sentence is the load-bearing half: an unterminated clause is joined to the
    next one as a single breath with no boundary between them.
    """
    text = " ".join(text.split())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?…" else text + "."


@dataclass
class Briefing:
    """What to say, and — separately — what to put on screen."""

    spoken: str
    paths: list[tuple[str, str]] = field(default_factory=list)

    def terminal_text(self) -> str:
        """The reference the spoken half deliberately leaves out."""
        if not self.paths:
            return self.spoken
        width = max(len(label) for label, _ in self.paths)
        lines = [f"  {label:<{width}}  {path}" for label, path in self.paths]
        return "status briefing\n" + "\n".join(lines)


def _parse_stamp(raw) -> datetime | None:
    """An ISO timestamp from configuration, or None if there is not one.

    A bare date with no time is accepted and means midnight, because "2026-09-23"
    is what gets typed when the time does not matter. Anything unparseable is
    treated as absent rather than raised: a mistyped date should cost the one
    sentence that needed it, not the briefing.
    """
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _incidents(cfg, critical: int, low: int, since: bool) -> str:
    """The one or two sentences about what has gone wrong, or the one saying
    nothing has.

    Assembled here rather than in a template because three things about it are
    decisions, not wording:

      a count of zero removes its sentence entirely — "there are also 0 low
      severity bugs I'd recommend reviewing" is the exact sentence the plural
      template would otherwise produce;

      "also" is only true if the other sentence was actually said, so it cannot
      be baked into the low-severity wording;

      "Since then" is only true if there is a previous briefing to be since, so
      it belongs to the first sentence and only when a timestamp exists.
    """
    if not critical and not low:
        key = "clear" if since else "clear_first"
        return _sentence(cfg.text(f"status.{key}", _DEFAULT_CLEAR))

    parts: list[str] = []
    if critical:
        key = "critical_one" if abs(critical) == 1 else "critical_many"
        parts.append(cfg.text(f"status.{key}", _DEFAULT_CRITICAL[key])
                     .format(n=critical))
    if low:
        key = "low_one" if abs(low) == 1 else "low_many"
        # "also" refers back to the critical sentence; empty when there is none.
        parts.append(cfg.text(f"status.{key}", _DEFAULT_LOW[key])
                     .format(n=low, also="also " if parts else ""))

    lead = "Since then, " if since else ""
    return " ".join(_sentence((lead if i == 0 else "") + part)
                    for i, part in enumerate(parts))


def build(cfg, now: datetime | None = None) -> Briefing:
    """Render the briefing from configuration and the clock.

    Never raises. A malformed template is a configuration mistake that must not
    cost Jay the answer, so the numbers are rendered into fallback wording
    instead and the mistake surfaces as prose rather than as silence.
    """
    now = now or datetime.now()
    operator = cfg.get("identity.operator", "Jay")

    state = str(cfg.get("status.state", _DEFAULT_STATE)).strip() or _DEFAULT_STATE
    critical = int(cfg.get("status.critical", 0) or 0)
    low = int(cfg.get("status.low_severity", 0) or 0)

    livefeed = str(cfg.get("status.livefeed", "") or "").strip()
    errorlog = str(cfg.get("status.errorlog", "") or "").strip()
    paths = [(label, p) for label, p in
             (("live feed", livefeed), ("error log", errorlog)) if p]

    stamp = _parse_stamp(cfg.get("status.last_update", ""))
    fields = {
        "greeting": greeting(now),
        "operator": operator,
        "last_update": spoken_datetime(stamp) if stamp else "",
        "state": state,
        "incidents": _incidents(cfg, critical, low, since=stamp is not None),
        # Dropped when there is nothing to look at as well as when there is
        # nowhere to look: "nothing has failed, and there is nothing waiting on
        # you — check the error log for further details" argues with itself.
        "feeds": (cfg.text("status.feeds", _DEFAULT_FEEDS)
                  if paths and (critical or low) else ""),
    }

    key = "message" if stamp else "message_first"
    template = cfg.text(f"status.{key}") or _DEFAULT_MESSAGE[key]
    try:
        spoken = template.format(**fields)
    except (KeyError, IndexError, ValueError):
        # An edited template naming a field that does not exist. Say the state and
        # the incidents anyway — a briefing that reports its own contents is worth
        # more than a veto, and wording this wrong is visible enough to get fixed.
        spoken = (f"{fields['greeting']}, {operator}. Current system state: "
                  f"{state}. {fields['incidents']} "
                  f"The status message template needs a look.")

    # One blank-line-free string. The templates are written across several lines
    # for legibility in the YAML, and every one of those breaks would otherwise
    # reach the unit splitter as whitespace.
    return Briefing(spoken=" ".join(spoken.split()), paths=paths)
