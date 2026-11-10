"""Speech synthesis and playback (Kokoro-82M, ONNX, CUDA).

Three techniques carry the latency budget here, in order of how much they matter:

1. Clause streaming. The response is cut at clause boundaries and synthesized one
   piece at a time, so playback starts on clause 1 while clause 2 is still on the
   GPU. First-audio latency becomes a function of the first clause, not the whole
   answer. The first clause is deliberately allowed to be very short; later ones
   are larger because by then we are ahead of the playhead and throughput matters
   more than latency.

2. A disk phrase cache. Every fixed line HIVE can say — presence responses,
   acknowledgements, fillers, refusals — is synthesized once and reused forever.
   These cost zero inference at speech time, which is what makes "HIVE, open
   chrome" answer in well under half a second.

3. Warm-up at boot. The first CUDA inference pays for kernel compilation and
   workspace allocation. That happens during startup, not during the first thing
   the user says.

Playback runs through one persistent output stream. Opening a stream per
utterance costs 50-150ms on PipeWire and adds a click at both ends.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import sounddevice as sd
import soundfile as sf

from ..events import (AUDIO_LEVEL, SPEAK_CHUNK, SPEAK_END, SPEAK_START, EventBus)
from .capture import Spectrum
from .prosody import seam_for, shape, to_spoken, unit_rate

KOKORO_SR = 24000

# Segmentation.
#
# v1 split on commas as well as sentence ends, which minimised time-to-first-audio
# and made the delivery choppy. v2 split on sentences and only broke inside one
# when it overran a budget — better, but still audibly blocky wherever a long
# sentence had to be broken, because of three things it did nothing about:
#
#   - WHERE it broke. Packing greedily to a character budget puts the seam at
#     whatever comma happens to be near the limit, which is very often in the
#     middle of a phrase. Kokoro gives every unit its own prosodic contour, so a
#     seam in a bad place is heard as a sentence ending where none exists.
#   - THE GAP AT THE SEAM. Every unit was faded to silence and back, then a fixed
#     pause was appended on top of the pause the comma had already produced. Two
#     pauses and a dip, at every seam.
#   - THE STEP IN LEVEL. Units are synthesized independently and do not come back
#     at matched loudness. A step at the seam reads as a cut even when the timing
#     is perfect.
#
# v3 addresses all three: seams are chosen by grammatical quality rather than
# position (_SEAMS), joins are short equal-power crossfades sized by what ended
# the unit (_Joiner), and units are level-matched to a running target.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Places a sentence may be broken, best first. A break after a semicolon or dash
# is nearly free; a break before a conjunction is natural; a bare comma is a last
# resort, because half of them are list separators where a falling contour is
# plainly wrong.
_SEAMS = [
    (3, re.compile(r"[;:]\s+")),
    (3, re.compile(r"\s+[—–]\s+")),
    (2, re.compile(r",\s+(?=(?:and|but|so|because|which|while|though|although|"
                   r"then|yet|or|since|unless|before|after)\b)", re.I)),
    (2, re.compile(r"\s+(?=(?:but|because|so that|which means|and then)\b)", re.I)),
    (1, re.compile(r",\s+")),
]

# Shorter than this and a unit is a fragment, not a phrase — speaking one alone
# sounds like an afterthought no matter how it is joined.
MIN_UNIT_CHARS = 28

FIRST_FADE_MS = 7          # onset of an utterance
LAST_FADE_MS = 12          # and its tail


@dataclass
class Unit:
    """One synthesis unit. `join` is the mark that ended it, or "".

    v4 replaced the three coarse seam classes with the mark itself, because a
    colon and a semicolon are not the same pause, and neither is a question mark
    and a full stop. What each one costs lives in prosody.SEAM.
    """

    text: str
    join: str = "."


def tail_mark(unit: str) -> str:
    """The punctuation that ended this unit, or "" for a break forced by length."""
    tail = unit.rstrip()[-1:] if unit.strip() else ""
    return tail if tail in ".!?…:;—," else ""


def _best_seam(sentence: str, lo: int, hi: int) -> int:
    """The best place to break `sentence` in [lo, hi], or -1.

    Highest grammatical quality wins; among equals the latest one does, because a
    later break means fewer units overall and every unit boundary costs something.
    """
    best_pri, best_at = -1, -1
    for pri, pat in _SEAMS:
        for m in pat.finditer(sentence):
            at = m.end()
            if lo <= at <= hi and (pri > best_pri or (pri == best_pri and at > best_at)):
                best_pri, best_at = pri, at
    return best_at


def split_units(text: str, first_budget: int = 140,
                rest_budget: int = 420) -> list[Unit]:
    """Split into synthesis units — whole sentences wherever possible.

    The budgets are ceilings, not targets: a sentence is emitted whole if it fits,
    and only broken when it does not. The first budget is smaller so a long answer
    still starts speaking promptly, while ordinary replies are never fragmented at
    all — which remains the best way to sound fluent, since a unit boundary is the
    one thing no amount of joining fully hides.
    """
    text = text.strip()
    if not text:
        return []

    units: list[Unit] = []
    for sentence in (s.strip() for s in _SENTENCE.split(text)):
        if not sentence:
            continue
        budget = first_budget if not units else rest_budget
        while len(sentence) > budget:
            # Search the back half of the budget window: breaking at the first
            # available seam would produce a run of very short units.
            lo = max(MIN_UNIT_CHARS, int(budget * 0.45))
            at = _best_seam(sentence, lo, budget)
            if at < 0:
                # No grammatical seam in range. Fall back to a word boundary —
                # never mid-word, which Kokoro reads as two invented words.
                at = sentence.rfind(" ", lo, budget)
                if at < 0:
                    break
            head, rest = sentence[:at].strip(), sentence[at:].strip()
            if not head or not rest:
                break
            units.append(Unit(head, tail_mark(head)))
            sentence, budget = rest, rest_budget
        if sentence:
            units.append(Unit(sentence, tail_mark(sentence)))

    # A trailing scrap is a fragment, not a sentence.
    #
    # Pop first, then append. Writing this as `units[-2] = f"... {units.pop()}"`
    # is wrong twice over: Python evaluates the right side first, so with exactly
    # two units the index becomes -2 on a one-element list (IndexError), and with
    # more than two it merges into the wrong element.
    if len(units) > 1 and len(units[-1].text) < MIN_UNIT_CHARS:
        tail = units.pop()
        units[-1] = Unit(f"{units[-1].text} {tail.text}", tail.join)
    return units


def _fade(audio: np.ndarray, head_ms: float, tail_ms: float) -> np.ndarray:
    if not len(audio):
        return audio
    out = np.asarray(audio, dtype=np.float32).copy()
    n = min(int(KOKORO_SR * head_ms / 1000), len(out) // 2)
    if n > 0:
        out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    n = min(int(KOKORO_SR * tail_ms / 1000), len(out) // 2)
    if n > 0:
        out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


class _Joiner:
    """Overlap-adds successive synthesis units into one continuous stream.

    Kokoro trims silence from every unit it renders, so consecutive units butt
    straight up against each other with a step at the seam. v2 handled that by
    fading both edges to zero and appending a fixed pause — which means every
    seam, including one in the middle of a phrase, contained a 12ms hole followed
    by up to 85ms of added silence, on top of whatever pause the punctuation had
    already produced. That is the blockiness.

    Here the seam is treated according to the mark that ended the unit — a full
    stop is not a semicolon and neither is an ellipsis — and according to what
    comes after it, because a speaker takes a beat before pivoting ("But", "That
    said") and none at all before a continuation. Levels are matched across units
    as well, since an unmatched step in loudness reads as a cut even when the
    timing is right.

    One unit is always held back, since joining needs both sides of the seam:
    the pause after a phrase is not decidable until you know what follows it.
    """

    def __init__(self, emit: Callable[[np.ndarray], None], sr: int = KOKORO_SR):
        self._emit = emit
        self._sr = sr
        self.reset()

    def reset(self) -> None:
        self._pending: np.ndarray | None = None
        self._pending_text = ""
        self._head_open = True      # pending's onset has not been joined to anything
        self._target_rms: float | None = None

    # -- level matching -----------------------------------------------------
    def _match(self, audio: np.ndarray) -> np.ndarray:
        """Nudge a unit toward the running loudness. Bounded to about ±2 dB.

        Bounded because the goal is to remove the step at the seam, not to
        flatten delivery — a sentence that is genuinely quieter should stay
        quieter. The target drifts toward the material for the same reason.
        """
        if not len(audio):
            return audio
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms < 1e-4:
            return audio
        if self._target_rms is None:
            self._target_rms = rms
            return audio
        gain = float(np.clip(self._target_rms / rms, 0.79, 1.26))
        self._target_rms = 0.82 * self._target_rms + 0.18 * rms
        return (audio * gain).astype(np.float32)

    # -- stream -------------------------------------------------------------
    def push(self, audio: np.ndarray, text: str = "") -> float:
        """Queue one unit and its text. Returns seconds of audio emitted.

        The text is needed as well as the audio because the seam is decided by
        the mark that ended the *previous* unit and the word that opens this one
        — neither of which is recoverable from samples.
        """
        audio = self._match(np.asarray(audio, dtype=np.float32))
        if self._pending is None:
            self._pending, self._pending_text = audio, text
            return 0.0

        prev, prev_text = self._pending, self._pending_text
        over_ms, pause_ms = seam_for(prev_text, text)
        n = int(self._sr * over_ms / 1000)
        n = min(n, len(prev) // 2, len(audio) // 2)

        if n > 0:
            # Equal power, so the crossfade holds loudness across the seam. A
            # linear one dips in the middle, which is audible as a soft click.
            t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
            seam = prev[-n:] * np.cos(t) + audio[:n] * np.sin(t)
            out = np.concatenate([prev[:-n], seam])
            rest, head_open = audio[n:], False
        else:
            out = _fade(prev, 0.0, FIRST_FADE_MS)
            rest, head_open = audio, True

        if self._head_open:
            out = _fade(out, FIRST_FADE_MS, 0.0)
        pause = int(self._sr * pause_ms / 1000)
        if pause:
            out = np.concatenate([out, np.zeros(pause, dtype=np.float32)])

        self._pending, self._pending_text, self._head_open = rest, text, head_open
        self._emit(out)
        return len(out) / self._sr

    def flush(self) -> float:
        if self._pending is None:
            return 0.0
        out = _fade(self._pending, FIRST_FADE_MS if self._head_open else 0.0,
                    LAST_FADE_MS)
        self._pending = None
        self._emit(out)
        return len(out) / self._sr


class Speaker:
    def __init__(self, cfg, bus: EventBus):
        self.cfg, self.bus = cfg, bus
        # A voice may be a stock name ("bf_emma") or a blend of several
        # ("bf_lily:0.55 + bf_emma:0.45"). Kokoro's style vectors are ordinary
        # arrays, so a weighted average of two of them is a third valid voice —
        # which is how we reach a register none of the four British females hits
        # on its own.
        self.voice = cfg.get("tts.voice", "bf_emma")
        self._voice_style: np.ndarray | None = None
        self.speed = float(cfg.get("tts.speed", 1.0))
        self.lang = cfg.get("tts.lang", "en-gb")
        self.want_cuda = cfg.get("tts.device", "cuda") == "cuda"
        # Prosody shaping. All three are knobs rather than constants because the
        # only honest way to tune delivery is by ear — `hive --voicelab` renders
        # the same lines with the pass on and off for exactly that.
        self._breathing = bool(cfg.get("tts.prosody.breathing", True))
        self._breath_words = int(cfg.get("tts.prosody.breath_max_words", 22))
        self._rate_span = float(cfg.get("tts.prosody.rate_span", 0.05))

        self._kokoro = None
        self._provider = "none"
        self._lock = threading.Lock()
        self._ready = threading.Event()

        self._cache_dir = cfg.cache_dir / "phrases"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, np.ndarray] = {}

        self._out: sd.OutputStream | None = None
        self._pcm: queue.Queue[np.ndarray | None] = queue.Queue()
        self._joiner = _Joiner(self._enqueue)
        self._residual = np.zeros(0, dtype=np.float32)
        self._speaking = threading.Event()
        self._speak_depth = 0
        self._speak_lock = threading.RLock()
        # Only one utterance may be assembling at a time.
        #
        # The latency filler and the answer it covers are produced on different
        # threads and share one joiner and one PCM queue, and the filler's guard
        # against overlapping is a check-then-act: it looks at whether speech has
        # started, then starts speaking. If the first clause of the answer became
        # ready in between, say_stream reset the joiner out from under a filler
        # that was mid-sentence and both utterances came out shredded.
        #
        # It survived on timing — the fillers were around a second and the window
        # is narrow — which is exactly the kind of bug that appears the moment
        # somebody adds a longer line to the bank. Serialising them costs the
        # answer a wait for a filler already in flight, which is not a cost at
        # all: there is one pair of speakers and they are busy.
        self._utterance = threading.Lock()
        self._interrupt = threading.Event()
        self._spectrum = Spectrum(int(cfg.get("ui.spectrum_bins", 128)), KOKORO_SR)
        self._on_speaking_change = None

    # -- lifecycle ----------------------------------------------------------
    def preload(self, on_ready=None) -> None:
        threading.Thread(target=self._load, args=(on_ready,),
                         name="hive-tts-load", daemon=True).start()

    def _load(self, on_ready=None) -> None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        t0 = time.monotonic()
        models = self.cfg.models / "kokoro"
        voices = str(models / "voices-v1.0.bin")
        gpu_model, cpu_model = models / "kokoro-v1.0.fp16-gpu.onnx", models / "kokoro-v1.0.onnx"

        opts = ort.SessionOptions()
        opts.log_severity_level = 4

        sess = None
        if self.want_cuda and gpu_model.exists():
            try:
                sess = ort.InferenceSession(str(gpu_model), sess_options=opts,
                                            providers=["CUDAExecutionProvider"])
                self._provider = "cuda"
            except Exception as exc:
                self.bus.log(f"TTS cuda session failed: {exc}", "warn")
                sess = None
        if sess is None:
            path = cpu_model if cpu_model.exists() else gpu_model
            opts.intra_op_num_threads = 6
            sess = ort.InferenceSession(str(path), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
            self._provider = "cpu"

        # kokoro-onnx's own GPU detection calls find_spec("onnxruntime-gpu"), which
        # can never match — module names cannot contain hyphens. Hence from_session.
        kokoro = Kokoro.from_session(sess, voices)
        self._kokoro = kokoro
        self._voice_style = self._resolve_voice(self.voice)
        try:
            kokoro.create("Ready.", voice=self._speak_voice(), speed=self.speed,
                          lang=self.lang)
        except Exception as exc:
            self.bus.log(f"TTS warmup failed: {exc}", "warn")
        self._ready.set()
        self.bus.log(
            f"TTS ready on {self._provider} in {(time.monotonic()-t0)*1000:.0f}ms", "audio"
        )
        self._open_stream()
        threading.Thread(target=self._build_cache, name="hive-tts-cache", daemon=True).start()
        if on_ready:
            on_ready()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def provider(self) -> str:
        return self._provider

    def wait_ready(self, timeout: float = 120.0) -> bool:
        return self._ready.wait(timeout)

    def set_on_speaking_change(self, fn) -> None:
        """Hook so the runtime can mute the mic exactly while audio is playing."""
        self._on_speaking_change = fn

    # -- output stream ------------------------------------------------------
    def _open_stream(self) -> None:
        device = self.cfg.get("audio.output.device", "default")
        try:
            self._out = sd.OutputStream(
                samplerate=KOKORO_SR, channels=1, dtype="float32",
                device=None if device in (None, "", "default") else device,
                blocksize=512, callback=self._playback, latency="low",
            )
            self._out.start()
        except Exception as exc:
            self.bus.log(f"output stream failed: {exc}", "error")
            self._out = None

    def _playback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        need = frames
        buf = self._residual
        while len(buf) < need:
            try:
                chunk = self._pcm.get_nowait()
            except queue.Empty:
                break
            if chunk is None:            # sentinel: end of this utterance
                break
            buf = np.concatenate([buf, chunk]) if len(buf) else chunk
        if len(buf) >= need:
            out, self._residual = buf[:need], buf[need:]
        else:
            out = np.zeros(need, dtype=np.float32)
            if len(buf):
                out[: len(buf)] = buf
            self._residual = np.zeros(0, dtype=np.float32)
        outdata[:, 0] = out

        # Feed the visualizer from HIVE's own voice so the columns move in time
        # with what is actually coming out of the speakers.
        if self._speaking.is_set():
            self._spectrum.push(out)
            spec = self._spectrum.compute()
            self.bus.publish(
                AUDIO_LEVEL, rms=float(np.sqrt(np.mean(out * out))),
                peak=float(np.max(np.abs(out))) if need else 0.0,
                spectrum=spec.tolist(), source="tts",
            )

    # -- voice resolution ---------------------------------------------------
    def _resolve_voice(self, spec: str) -> np.ndarray | None:
        """Turn a blend spec into a style vector. Returns None for a plain name."""
        if self._kokoro is None or ":" not in spec:
            return None
        total, acc = 0.0, None
        for part in re.split(r"[+,]", spec):
            name, _, weight = part.partition(":")
            name = name.strip()
            if name not in self._kokoro.voices:
                self.bus.log(f"unknown voice in blend: {name}", "warn")
                return None
            w = float(weight) if weight.strip() else 1.0
            vec = np.asarray(self._kokoro.voices[name], dtype=np.float32) * w
            acc = vec if acc is None else acc + vec
            total += w
        if acc is None or total <= 0:
            return None
        self.bus.log(f"voice blend: {spec}", "audio")
        return acc / total

    def _speak_voice(self):
        """What to hand Kokoro: a blended array if we have one, else the name."""
        return self._voice_style if self._voice_style is not None else self.voice

    # -- synthesis ----------------------------------------------------------
    def _key(self, text: str, speed: float | None = None) -> str:
        raw = f"{self.voice}|{speed if speed is not None else self.speed}|{self.lang}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def synth(self, text: str, use_cache: bool = True,
              speed: float | None = None) -> np.ndarray:
        """Synthesize one clause. Returns float32 @24k.

        With `use_cache` this both reads *and* writes. It used to only read,
        which quietly made the whole speculative path pointless: the terminal
        speculator renders a holding line while the request is still being typed
        precisely so the turn finds finished audio, and a synth that discards
        what it generated meant every one of those renders was thrown away and
        paid for twice.
        """
        text = to_spoken(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        rate = self.speed if speed is None else float(speed)
        key = self._key(text, rate)
        if use_cache:
            hit = self._mem.get(key)
            if hit is not None:
                return hit
            path = self._cache_dir / f"{key}.wav"
            if path.exists():
                try:
                    data, _ = sf.read(path, dtype="float32")
                    self._mem[key] = data
                    return data
                except Exception:
                    pass
        if not self._ready.wait(60.0) or self._kokoro is None:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            # Check again now that we hold the lock. The speculator renders a
            # holding line while the request is still being typed, and pressing
            # Enter mid-render used to queue a second identical synthesis behind
            # the first — paying twice and arriving late. Now the turn simply
            # waits out the render that is already running and takes its result.
            if use_cache:
                hit = self._mem.get(key)
                if hit is not None:
                    return hit
            samples, _sr = self._kokoro.create(
                text, voice=self._speak_voice(), speed=rate, lang=self.lang
            )
        audio = np.asarray(samples, dtype=np.float32)
        if use_cache and len(audio):
            self._remember(key, text, audio)
        return audio

    def _remember(self, key: str, spoken: str, audio: np.ndarray) -> None:
        """Hold a rendered line in RAM and on disk so it is free next time."""
        if len(self._mem) > 512:
            # Bounded. Streamed answer units never come through here — they are
            # synthesized with use_cache=False — so this only ever holds fixed
            # lines, acknowledgements and holding lines.
            for stale in list(self._mem)[:128]:
                self._mem.pop(stale, None)
        self._mem[key] = audio
        try:
            sf.write(self._cache_dir / f"{key}.wav", audio, KOKORO_SR)
        except Exception:
            pass

    def _cache_to_disk(self, text: str, audio: np.ndarray,
                       speed: float | None = None) -> None:
        try:
            sf.write(self._cache_dir / f"{self._key(to_spoken(text), speed)}.wav",
                     audio, KOKORO_SR)
        except Exception:
            pass

    def _build_cache(self) -> None:
        """Pre-synthesize every fixed line, once, ever.

        Runs in the background after boot. Until it finishes, those lines fall
        back to live synthesis, so a cold first run is slower but never broken.

        Cached per *unit*, not per line. Speech goes out one unit at a time and
        the cache is keyed on what is handed to Kokoro, so a two-sentence line
        rendered whole here produced a key nothing would ever look up — those
        entries were written, never read, and the line paid full synthesis every
        time it was spoken. Going through the same planner the speaking path
        uses is what keeps the two from drifting apart again.
        """
        banks = [
            "presence.default", "presence.late_night", "presence.early_morning",
            "presence.repeat", "ack.opening", "ack.done", "ack.saved",
            "filler.short", "filler.thinking", "confirm.system_query",
            "confirm.shell_exec", "confirm.outside_vault", "confirm.granted",
            "confirm.denied", "veto.destructive", "veto.ungrounded",
            "veto.not_found", "veto.error", "greeting.cold", "greeting.returning",
            "farewell", "misheard",
            # The watchdog line, spoken when a turn has been silent for three and
            # a half seconds. It was missing from this list, which meant the one
            # line guaranteed to be needed while the machine is already busy was
            # also the one line that had to be synthesized on the spot.
            "hold",
        ]
        lines: list[str] = []
        for bank in banks:
            lines += [ln for ln in self.cfg.bank(bank) if "{" not in ln]
        # Templated banks are rendered per target, so the fixed pass above skips
        # them. Render the openings for the applications actually opened most —
        # "Opening Spotify" is a fixed string once the app's name is known, and
        # pre-rendering the habitual ones keeps the launch path at zero latency
        # from the first request of the session rather than the second.
        lines += self._habitual_openings()

        made = 0
        for line in lines:
            for text, rate in self.plan(line):
                key = self._key(to_spoken(text), rate)
                if (self._cache_dir / f"{key}.wav").exists():
                    continue
                try:
                    audio = self.synth(text, use_cache=False, speed=rate)
                    if len(audio):
                        self._cache_to_disk(text, audio, rate)
                        made += 1
                except Exception:
                    continue
        if made:
            self.bus.log(f"phrase cache built (+{made})", "audio")

    def _habitual_openings(self) -> list[str]:
        from ..cognition.phrasing import fill

        counts = self.cfg.get("_app_counts", None)
        if not isinstance(counts, dict) or not counts:
            return []
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        out: list[str] = []
        for name, _n in top:
            label = name.strip().title()
            out += [fill(t, target=label) for t in self.cfg.bank("ack.opening")]
        return out

    # -- speaking -----------------------------------------------------------
    # Reference-counted, not a boolean. A latency filler and the streamed answer
    # it covers overlap: the filler's say() finishes while the answer is still
    # playing. With a boolean, that inner _end() would clear the speaking flag
    # and unmute the microphone mid-answer, and HIVE would start transcribing
    # itself. The count only reaches zero when the last overlapping speech ends.
    def _begin(self) -> None:
        with self._speak_lock:
            self._speak_depth += 1
            if self._speak_depth == 1:
                self._speaking.set()
                self.bus.publish(SPEAK_START)
                if self._on_speaking_change:
                    self._on_speaking_change(True)

    def _end(self) -> None:
        with self._speak_lock:
            if self._speak_depth == 0:
                return
            self._speak_depth -= 1
            if self._speak_depth > 0:
                return
            self._speaking.clear()
        self.bus.publish(SPEAK_END)
        if self._on_speaking_change:
            self._on_speaking_change(False)

    def _enqueue(self, audio: np.ndarray) -> float:
        if len(audio):
            self._pcm.put(audio)
        return len(audio) / KOKORO_SR

    def _drain(self, seconds: float) -> None:
        """Block until roughly `seconds` of queued audio has played."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._interrupt.is_set():
                return
            time.sleep(0.01)

    def _publish_chunk(self, text: str, first: bool) -> None:
        """Announce a unit as it goes to the speakers, for the terminal mirror."""
        try:
            self.bus.publish(SPEAK_CHUNK, text=text, first=first)
        except Exception:
            pass

    # -- planning -----------------------------------------------------------
    def plan(self, text: str) -> list[tuple[str, float]]:
        """Split a complete utterance into (unit, speaking rate) pairs.

        The single place that decides both, so the phrase cache and the speaking
        path can never disagree about what will be handed to Kokoro. Rate varies
        with position because that is what human phrasing does — an utterance
        opens slower, runs on a shade quicker, and stretches into its last
        phrase — and because the cache is keyed on rate, a drift of one
        thousandth between here and playback would silently cost a full
        synthesis on every fixed line.
        """
        units = split_units(
            shape(text, max_words=self._breath_words, breath=self._breathing),
            first_budget=int(self.cfg.get("tts.first_unit_chars", 140)),
            rest_budget=int(self.cfg.get("tts.unit_chars", 420)),
        )
        n = len(units)
        out: list[tuple[str, float]] = []
        for i, unit in enumerate(units):
            position = ("only" if n == 1 else
                        "open" if i == 0 else
                        "close" if i == n - 1 else "mid")
            out.append((unit.text,
                        unit_rate(self.speed, position, unit.text, self._rate_span)))
        return out

    def say(self, text: str, cached: bool = True) -> None:
        """Speak a complete string, blocking until it finishes."""
        plan = self.plan(text)
        if not plan:
            return
        with self._utterance:
            self._say_locked(plan, cached)

    def _say_locked(self, plan: list[tuple[str, float]], cached: bool) -> None:
        self._interrupt.clear()
        self._begin()
        total = 0.0
        first = True
        try:
            for unit_text, rate in plan:
                if self._interrupt.is_set():
                    break
                # synth() persists it when cached — no second write needed here.
                audio = self.synth(unit_text, use_cache=cached, speed=rate)
                self._publish_chunk(unit_text, first)
                first = False
                # Joining happens here, not in synth(), so cached audio gets the
                # same treatment at the seams as freshly-generated audio.
                total += self._joiner.push(audio, unit_text)
            if not self._interrupt.is_set():
                total += self._joiner.flush()
            self._drain(total + 0.10)
        finally:
            self._joiner.reset()
            self._end()

    def render_offline(self, text: str) -> np.ndarray:
        """Render a string exactly as playback would, without playing it.

        Used by the voice lab. A voice that sounds fine as one blob and choppy in
        production is not a useful audition, so this has to go through the same
        segmentation and the same joiner rather than concatenating units.
        """
        pieces: list[np.ndarray] = []
        joiner = _Joiner(pieces.append)
        for unit_text, rate in self.plan(text):
            joiner.push(self.synth(unit_text, use_cache=False, speed=rate), unit_text)
        joiner.flush()
        return (np.concatenate(pieces) if pieces
                else np.zeros(0, dtype=np.float32))

    def say_line(self, bank: str) -> str:
        """Speak a persona line straight from cache. The zero-latency path."""
        line = self.cfg.line(bank)
        if line:
            self.say(line, cached=True)
        return line

    def say_stream(self, chunks: Iterable[str], on_first_audio=None) -> str:
        """Speak an incrementally-arriving response.

        Accumulates tokens, synthesizes as soon as a clause boundary is reached,
        and starts playback on the first one. This is what keeps first-audio in
        the low hundreds of milliseconds for model-generated answers.

        `on_first_audio` fires the moment the first clause is queued for
        playback. Without it, callers can only time the whole call — which
        includes waiting out every second of speech — and a fast turn reads as
        an eight-second one.
        """
        buf, spoken, total = "", [], 0.0
        first = True
        held = False
        self._interrupt.clear()

        # `speaking` is NOT set here. It is set on the first enqueue, below.
        # Marking it up front would mean the speaker claims to be speaking during
        # the entire wait for the first token — which suppresses the latency
        # filler exactly when the answer is slow enough to need one, and mutes
        # the microphone seconds before any sound comes out.
        def begin_once() -> None:
            nonlocal first, held
            if first:
                first = False
                # Wait for a filler to finish here, at the first clause, rather
                # than at the top of the method. Tokens keep arriving from the
                # model throughout, so the wait costs only the audio it is
                # waiting for — and taking the lock any earlier would hold it
                # across the whole wait for the first token, which is the exact
                # window the filler exists to fill.
                self._utterance.acquire()
                held = True
                self._joiner.reset()
                self._begin()
                if on_first_audio:
                    on_first_audio()

        try:
            first_budget = int(self.cfg.get("tts.first_unit_chars", 140))
            rest_budget = int(self.cfg.get("tts.unit_chars", 420))
            emitted = 0
            for chunk in chunks:
                if self._interrupt.is_set():
                    break
                buf += chunk
                budget = first_budget if first else rest_budget
                # Prosody shaping runs on the growing buffer rather than on each
                # finished unit, because the things it fixes — an enumerator, a
                # clause that overruns a breath — span the boundaries the
                # splitter is about to choose. Every stage of it is idempotent
                # for exactly this reason: the buffer passes through here once
                # per token and must not accumulate rewrites of its own output.
                shaped = shape(buf, max_words=self._breath_words,
                               breath=self._breathing, terminal=False)
                parts = split_units(shaped, first_budget=budget, rest_budget=budget)
                # Hold the tail back — it may still be growing. A unit is only
                # spoken once a later one exists, which proves it is complete.
                while len(parts) > 1:
                    unit = parts.pop(0)
                    rate = unit_rate(self.speed, "open" if emitted == 0 else "mid",
                                     unit.text, self._rate_span)
                    audio = self.synth(unit.text, use_cache=False, speed=rate)
                    self._publish_chunk(unit.text, first)
                    begin_once()
                    total += self._joiner.push(audio, unit.text)
                    spoken.append(unit.text)
                    emitted += 1
                    # Trailing space, deliberately. The units come back stripped,
                    # and the next token off the model is appended raw — so
                    # rebuilding the buffer without one leaves the seam depending
                    # on whether that token happens to arrive with a leading
                    # space. Ollama's usually do, which is the only reason this
                    # never showed up as two words run together.
                    buf = " ".join(p.text for p in parts) + " "
                    parts = split_units(buf, first_budget=rest_budget,
                                        rest_budget=rest_budget)
            tail = shape(buf, max_words=self._breath_words, breath=self._breathing)
            if tail and not self._interrupt.is_set():
                # The closing phrase, so it gets phrase-final lengthening — the
                # strongest cue there is that an utterance has actually ended,
                # rather than been cut off.
                rate = unit_rate(self.speed, "only" if emitted == 0 else "close",
                                 tail, self._rate_span)
                audio = self.synth(tail, use_cache=False, speed=rate)
                self._publish_chunk(tail, first)
                begin_once()          # short answer: one unit, never hit the loop
                total += self._joiner.push(audio, tail)
                spoken.append(tail)
            if not self._interrupt.is_set():
                total += self._joiner.flush()
            self._drain(total + 0.10)
        finally:
            # Only touch the joiner if this call ever owned it. An answer that
            # produced no audio at all must not reset one belonging to a filler
            # that is still speaking — which is the same collision this lock
            # exists to prevent, arriving through the error path.
            if held:
                self._joiner.reset()
                self._end()
                self._utterance.release()
        return " ".join(spoken)

    # -- control ------------------------------------------------------------
    def interrupt(self) -> None:
        """Stop immediately and discard pending audio (Esc, or right-click)."""
        self._interrupt.set()
        try:
            while True:
                self._pcm.get_nowait()
        except queue.Empty:
            pass
        self._residual = np.zeros(0, dtype=np.float32)
        # Drop the held-back unit too, or the next utterance would open with the
        # tail of the one that was just cancelled.
        self._joiner.reset()
        # Collapse the whole count: an interrupt cancels every overlapping
        # speech at once, not just the innermost one.
        with self._speak_lock:
            was_speaking = self._speak_depth > 0
            self._speak_depth = 0
            self._speaking.clear()
        if was_speaking:
            self.bus.publish(SPEAK_END)
            if self._on_speaking_change:
                self._on_speaking_change(False)

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def closed(self) -> bool:
        """Has the output stream gone away?

        Asked by work that outlives its own turn — a background write-up waits
        for a gap in the conversation before announcing itself, and the gap it
        is waiting for may turn out to be the session ending.
        """
        return self._out is None

    def set_voice(self, voice: str) -> bool:
        """Swap voice at runtime. Accepts a stock name or a blend spec.

        Cache keys include the voice spec, so a swap never collides with audio
        already rendered in the previous voice.
        """
        if self._kokoro is None:
            return False
        if ":" in voice:
            style = self._resolve_voice(voice)
            if style is None:
                return False
            self._voice_style = style
        elif voice in getattr(self._kokoro, "voices", {}):
            self._voice_style = None
        else:
            return False
        self.voice = voice
        self._mem.clear()
        threading.Thread(target=self._build_cache, daemon=True).start()
        return True

    def available_voices(self) -> list[str]:
        if self._kokoro is None:
            return []
        return sorted(self._kokoro.voices)

    def close(self) -> None:
        self.interrupt()
        if self._out is not None:
            try:
                self._out.stop()
                self._out.close()
            finally:
                self._out = None
