"""Live microphone diagnostic: `hive --miccheck`.

Runs the real input path — capture, endpointing, recognition, wake gate, intent —
against your actual microphone, and prints what it sees. Nothing speaks, so there
is no feedback loop and nothing to mute.

This is the tool for "HIVE can see my voice but never answers". It shows exactly
which stage drops the turn:

    no LEVEL bars      the microphone is not being captured at all
    bars but no VAD    speech is arriving but not being endpointed
    VAD but no text    the recogniser is failing
    text but no wake   it heard you, but not your name
    wake but no intent it understood you and matched nothing

Ctrl-C to stop.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from . import _cuda


def run() -> int:
    _cuda.preload()
    from .audio.asr import Recognizer
    from .audio.capture import MicCapture
    from .audio.vad import Endpointer
    from .cognition.intents import classify, detect_wake
    from .config import load
    from .events import AUDIO_LEVEL, EventBus

    cfg, bus = load(), EventBus()

    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        print(f"\n\033[96mHIVE mic check\033[0m   input: {dev['name']}  "
              f"({dev['default_samplerate']:.0f}Hz native)")
    except Exception as exc:
        print(f"\n\033[91mno input device: {exc}\033[0m")
        return 1
    print(f"threshold={cfg.get('vad.threshold')}  "
          f"min_speech={cfg.get('vad.min_speech_ms')}ms  "
          f"hangover={cfg.get('vad.min_silence_ms')}ms")
    print("─" * 74)

    rec = Recognizer(cfg, bus)
    rec.preload()
    print("loading recogniser...", end=" ", flush=True)
    if not rec.wait_ready(180):
        print("\033[91mfailed\033[0m")
        return 1
    print("ready.\n\033[96mSpeak now — try \"HIVE, are you there?\"\033[0m  (Ctrl-C to stop)\n")

    peak = {"v": 0.0, "n": 0}

    def on_level(evt) -> None:
        if evt.payload.get("source") != "mic":
            return
        peak["v"] = max(peak["v"], float(evt.payload.get("peak", 0.0)))
        peak["n"] += 1

    bus.subscribe(AUDIO_LEVEL, on_level)

    cap = MicCapture(cfg, bus)
    cap.start()

    count = {"n": 0}

    def on_utterance(utt) -> None:
        count["n"] += 1
        t0 = time.monotonic()
        text = rec.transcribe(utt.audio)
        ms = (time.monotonic() - t0) * 1000
        print(f"\033[96m▸ utterance {count['n']}\033[0m  {utt.duration:.2f}s  "
              f"(asr {ms:.0f}ms)")
        if not text:
            print("   \033[93mrecogniser returned nothing\033[0m — too quiet, or not speech\n")
            return
        print(f"   heard   : \033[97m{text}\033[0m")
        woke, rest = detect_wake(text, cfg)
        if woke:
            intent = classify(rest)
            slots = {k: v for k, v in intent.slots.items() if v}
            print(f"   wake    : \033[92myes\033[0m  -> \033[97m{rest or '(nothing after the name)'}\033[0m")
            print(f"   intent  : \033[92m{intent.name}\033[0m"
                  + (f"  {slots}" if slots else ""))
        else:
            print("   wake    : \033[93mNO — HIVE would ignore this\033[0m")
            print("             say the name first, or it only counts inside the "
                  "15s follow-up window")
        print()

    ep = Endpointer(cfg, bus, cap, on_utterance)
    ep.start()

    # Heartbeat, so silence is distinguishable from a dead capture path.
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(3.0):
            if count["n"] == 0:
                bar = int(min(1.0, peak["v"]) * 40)
                lvl = "█" * bar + "·" * (40 - bar)
                state = "\033[92mhearing you\033[0m" if peak["v"] > 0.02 else \
                        "\033[93msilence\033[0m"
                print(f"   level [{lvl}] peak={peak['v']:.3f}  {state}", end="\r", flush=True)
            peak["v"] = 0.0

    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        ep.stop()
        cap.stop()
        print(f"\n\n{count['n']} utterance(s) detected.")
        if count["n"] == 0:
            print("\033[93mNothing was endpointed.\033[0m If the level bar moved, the "
                  "microphone is fine and the VAD is the problem — run "
                  "`--selftest` and check the endpointer line.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
