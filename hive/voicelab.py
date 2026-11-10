"""Voice audition: `hive --voicelab`.

Renders the same lines in every British-female candidate and several blends,
plays them in order, and writes the WAVs to `assets/voicelab/` so they can be
compared again without re-running.

Blends exist because none of the four stock British voices sits exactly where we
want. Kokoro's style vectors are plain arrays, so a weighted average of two is a
third valid voice — that is how you reach a younger register while keeping the
composure of the more mature one.

Pick a line, then set it in config/hive.toml:

    [tts]
    voice = "bf_lily:0.6 + bf_emma:0.4"

or say "HIVE, switch to <name>" while it is running.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from . import _cuda

# Deliberately prosody-heavy: a short answer, a clause with a comma, a question,
# and a number. Flat single sentences make every voice sound the same.
SAMPLE = (
    "Right here. I've opened the Claude chat and a new Google Doc for you, "
    "and the eJPT note is in the vault. That's a longer answer — want the full version?"
)

CANDIDATES: list[tuple[str, str]] = [
    ("bf_emma",                    "stock — current default, mature and warm"),
    ("bf_alice",                   "stock — brighter, younger"),
    ("bf_isabella",                "stock — mid, rounder"),
    ("bf_lily",                    "stock — lightest, youngest"),
    ("bf_lily:0.6 + bf_emma:0.4",  "blend — young, with Emma's composure"),
    ("bf_alice:0.55 + bf_emma:0.45", "blend — bright but grounded"),
    ("bf_lily:0.5 + bf_isabella:0.5", "blend — young and rounded"),
    ("bf_alice:0.4 + bf_lily:0.35 + bf_emma:0.25", "blend — three-way, youngest composed"),
]


def run(play: bool = True) -> int:
    _cuda.preload()
    from .audio.tts import Speaker, split_units
    from .config import load
    from .events import EventBus

    import soundfile as sf

    cfg, bus = load(), EventBus()
    sp = Speaker(cfg, bus)
    sp.preload()
    print("\n\033[96mHIVE voice lab\033[0m   loading Kokoro...", end=" ", flush=True)
    if not sp.wait_ready(180):
        print("\033[91mfailed\033[0m")
        return 1
    print(f"ready on {sp.provider}.\n")

    out_dir = cfg.root / "assets" / "voicelab"
    out_dir.mkdir(parents=True, exist_ok=True)
    speed = float(cfg.get("tts.speed", 1.0))
    print(f"speed={speed}  sample: {SAMPLE[:64]}...\n" + "─" * 74)

    for i, (spec, note) in enumerate(CANDIDATES, 1):
        if not sp.set_voice(spec):
            print(f"  \033[91m{i}. {spec} — unavailable\033[0m")
            continue
        # Render exactly the way the runtime would: same segmentation, same
        # crossfades, pauses and level matching. A voice that sounds fine as one
        # blob and choppy in production is not a useful audition.
        units = split_units(SAMPLE)
        audio = sp.render_offline(SAMPLE)

        safe = spec.replace(" ", "").replace(":", "-").replace("+", "_").replace(".", "")
        path = out_dir / f"{i:02d}_{safe}.wav"
        sf.write(path, audio, 24000)
        print(f"  \033[96m{i}.\033[0m {spec:44s} {note}")
        print(f"     {len(audio)/24000:.1f}s  ->  {path.name}   ({len(units)} units)")

        if play:
            sp.say(f"Candidate {i}.", cached=False)
            time.sleep(0.15)
            sp._begin()
            sp._enqueue(audio)
            sp._drain(len(audio) / 24000 + 0.35)
            sp._end()
            time.sleep(0.45)

    print("─" * 74)
    print(f"\nWAVs in {out_dir}")
    print("Set your pick in config/hive.toml:\n")
    print('    [tts]\n    voice = "bf_lily:0.6 + bf_emma:0.4"\n')

    _prosody_ab(sp, cfg, out_dir, play)

    sp.close()
    return 0


# What a language model actually returns when it explains something: a numbered
# list, a hyphen standing in for a dash, and a sentence long enough to run out of
# air. Every one of those is a specific, audible failure in Kokoro, which is why
# this is the comparison sample rather than a nice paragraph.
PROSODY_SAMPLE = (
    "Smoothing a voice agent comes down to three things: "
    "1. Prosody - pitch, tempo and where the pauses fall. "
    "2. Emotional modelling, which is sentiment analysis deciding tone and pace from context. "
    "3. Contextual awareness of dialogue flow and speaker intent so the delivery matches "
    "what was actually meant rather than what was literally said. "
    "The first one is the only one you can control from the text."
)


def _prosody_ab(sp, cfg, out_dir, play: bool) -> None:
    """Render the same passage with the prosody pass on and off.

    Delivery cannot be tuned by reading the diff. This is the only way to answer
    "is the breathing helping" honestly, so the knobs in `[tts.prosody]` have
    something to be set against.
    """
    from .audio.prosody import shape

    import soundfile as sf

    sp.set_voice(cfg.get("tts.voice", "bf_emma"))
    print("\n\033[96mprosody pass\033[0m   same passage, shaping off then on")
    print("─" * 74)
    print(f"  raw   : {PROSODY_SAMPLE[:96]}...")
    print(f"  shaped: {shape(PROSODY_SAMPLE)[:96]}...")
    print()

    for label, breathing, span in (("off", False, 0.0), ("on", True, None)):
        was_breath, was_span = sp._breathing, sp._rate_span
        sp._breathing = breathing
        if span is not None:
            sp._rate_span = span
        # With shaping off, hand Kokoro the text exactly as the model wrote it —
        # which is the whole point of the comparison.
        if breathing:
            audio = sp.render_offline(PROSODY_SAMPLE)
        else:
            audio = sp.synth(PROSODY_SAMPLE, use_cache=False)
        sp._breathing, sp._rate_span = was_breath, was_span

        path = out_dir / f"prosody_{label}.wav"
        sf.write(path, audio, 24000)
        print(f"  shaping {label:3s}  {len(audio)/24000:5.1f}s  ->  {path.name}")
        if play:
            sp.say(f"Shaping {label}.", cached=False)
            time.sleep(0.15)
            sp._begin()
            sp._enqueue(audio)
            sp._drain(len(audio) / 24000 + 0.35)
            sp._end()
            time.sleep(0.5)

    print("\n  Tune in config/hive.toml under [tts.prosody]:")
    print("      breathing = true / false")
    print("      breath_max_words = 22        # lower = more pauses")
    print("      rate_span = 0.05             # 0 = flat delivery\n")


if __name__ == "__main__":
    sys.exit(run(play="--no-play" not in sys.argv))
