"""End-to-end turn test: `hive --dryrun`.

Drives complete turns through the real pipeline — recogniser, wake gate, intent
grammar, both hemispheres, the callosum, and speech — without a microphone.
Each phrase is synthesized in HIVE's own voice, resampled to 16k, and handed to
the recogniser exactly as the endpointer would hand it over.

Launching is stubbed. Everything else is real, including writes to the vault, so
this is the closest you can get to a conversation without having one.

Reports the latency of every stage against the budget, which is the point: it
tells you *where* a slow turn went slow.
"""

from __future__ import annotations

import time

import numpy as np

from . import _cuda

PHRASES = [
    "HIVE, are you there?",
    "HIVE, what time is it?",
    # v8: the status briefing. Deterministic and spoken straight from
    # configuration — the check here is that a message written to be read
    # survives being said, which is a different question from whether the
    # grammar matched it. Verbatim from Jay's request.
    "HIVE, can you give me a current status update?",
    "HIVE, open up a new Claude chat for me.",
    "HIVE, open up a new Google Docs window for me on Google Chrome.",
    "HIVE, make a note that the eJPT exam is on Friday.",
    "HIVE, how much disk space do I have?",
    # The request that used to freeze: understood, unfulfillable, must produce a
    # reason and a concrete alternative rather than silence.
    "HIVE, play my deep focus playlist.",
    "HIVE, what folders do you have access to?",
    # v2: the compound case — open something AND put a question into it.
    "HIVE, open a new Claude chat asking what is the mathematical equation "
    "behind buffer formalism.",
    # v2: genuinely multi-step; every step must be reported on.
    "HIVE, open spotify and then make a note that the A plus exam is on Friday.",
    # v2: job search — must open filtered searches and name the network boundary.
    "HIVE, check indeed and ziprecruiter for overnight help desk in San Diego.",
    "HIVE, are you there?",
    "HIVE, what should I focus on after the eJPT?",
    # v5: a launch verb pointed at a subject. Must NOT open an application —
    # this exact phrasing opened Seahorse in a real session.
    "HIVE, pull up some articles on human sounding voice agents.",
    # v6: the same shape of request, now that there is a network to answer it
    # with. Goes to arXiv and the open web for real — the launcher is stubbed
    # here, the network deliberately is not, because whether HIVE can actually
    # reach the internet is exactly what a dry run should find out.
    "HIVE, find me research articles on neural networks.",
    # v6: open a site's own search, already filled in. The URL is what makes
    # "type it into the search bar" work without an automation stack.
    "HIVE, open YouTube and search for hardcore gameplay.",
    # v5: the request that used to be answered with a promise and no file.
    # Deliberately last, so it has a real conversation to write up.
    "HIVE, create a markdown file with everything we talked about, in detail.",
]


def _to16k(samples: np.ndarray, sr: int = 24000) -> np.ndarray:
    a = np.asarray(samples, dtype=np.float32)
    n = int(len(a) * 16000 / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def run(phrases: list[str] | None = None) -> int:
    _cuda.preload()
    from .config import load
    from .events import EventBus, Stopwatch
    from .runtime import Runtime

    cfg, bus = load(), EventBus()
    rt = Runtime(cfg, bus)

    # Stub the launcher: resolve for real, never exec. Otherwise a dry run opens
    # four browser windows on the operator's desktop.
    launched: list[str] = []
    real_resolve = rt.launcher.resolve

    def fake_launch(target):
        launched.append(f"{target.kind}:{target.label} -> {' '.join(target.argv)}")
        return True, target.label
    rt.launcher.launch = fake_launch  # type: ignore[method-assign]

    print("\n\033[96mHIVE dry run\033[0m  (launches stubbed, vault writes are real)")
    print("─" * 74)

    rt.ltm.ensure()
    rt.subjective.load()
    rt.recognizer.preload()
    rt.speaker.preload()
    if not (rt.recognizer.wait_ready(180) and rt.speaker.wait_ready(180)):
        print("engines failed to load")
        return 1
    rt.speaker.set_on_speaking_change(lambda s: None)   # no mic to mute
    if rt.llm.available():
        rt.llm.preload()
        time.sleep(2.0)
    else:
        print("\033[93m ! Ollama unreachable — conversational turns will be empty\033[0m")

    rows: list[tuple[str, float, str, str]] = []
    for phrase in (phrases or PHRASES):
        audio16 = _to16k(rt.speaker.synth(phrase, use_cache=False))
        sw = Stopwatch(bus, "dry")

        heard = rt.recognizer.transcribe(audio16)
        sw.mark("asr")

        from .cognition.intents import classify, detect_wake
        woke, rest = detect_wake(heard, cfg)
        payload = rest if woke else heard
        intent = classify(payload)

        t0 = time.monotonic()
        outcome = rt.callosum.handle(payload, sw)
        turn_ms = (time.monotonic() - t0) * 1000

        # Time to first audio is the number that matters. Total turn time
        # includes waiting out the whole spoken answer, which is dominated by how
        # long the reply is and says nothing about responsiveness.
        first = next((ms for n, ms in sw.marks if n == "first_audio"), None)
        rows.append((phrase, first if first is not None else turn_ms,
                     intent.name, outcome.spoke))

        print(f"\n\033[96m▸\033[0m {phrase}")
        print(f"   heard   : {heard}")
        print(f"   intent  : {intent.name}"
              + (f"  slots={ {k: v for k, v in intent.slots.items() if v} }"
                 if intent.slots else ""))
        print(f"   said    : {outcome.spoke[:190]}")
        print(f"   timing  : {sw.breakdown()}"
              f"   (spoken over {turn_ms/1000:.1f}s)")

    # Session close, exercising consolidation for real.
    print("\n" + "─" * 74)
    # A requested write-up runs on its own thread and outlives its turn, so the
    # real shutdown waits for it. Do the same here or the dry run exits while
    # the file it is meant to be testing is still being written.
    print("waiting for any write-up in flight...")
    rt.objective.wait_for_writes()
    print("consolidating to vault...")
    rt.consolidate()

    print("\n\033[96mstubbed launches\033[0m")
    for line in launched:
        print(f"   {line}")

    budget = float(cfg.get("latency.target_first_audio_ms", 800))
    over = [(p, ms) for p, ms, _i, _s in rows if ms > budget]
    print(f"\n\033[96mlatency\033[0m  budget {budget:.0f}ms to first audio")
    for phrase, ms, intent, _ in rows:
        flag = "\033[93m OVER\033[0m" if ms > budget else "\033[92m  ok \033[0m"
        print(f"  {flag} {ms:6.0f}ms  {intent:14s} {phrase[:44]}")
    print()
    if over:
        print(f"\033[93m{len(over)}/{len(rows)} turns over budget\033[0m\n")
    else:
        print(f"\033[92mall {len(rows)} turns within budget\033[0m\n")

    rt.speaker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
