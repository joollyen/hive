"""Speech recognition (faster-whisper / CTranslate2 on CUDA).

Settings here are tuned for command-and-control latency rather than transcript
quality on long-form audio:

  beam_size=1                  greedy. Beam search costs ~80ms and buys nothing
                               measurable on utterances this short.
  condition_on_previous_text   off. With it on, Whisper occasionally locks into a
                               repetition loop after a noisy segment and emits the
                               same clause forever.
  vad_filter                   off. The endpointer already did this; running
                               Whisper's own VAD over pre-segmented audio is pure
                               duplicated cost.
  hotwords                     biases the decoder toward HIVE's actual working
                               vocabulary — without it "eJPT" comes back as
                               "EGPT" and "Impacket" as "in packet".
"""

from __future__ import annotations

import re
import threading
import time

import numpy as np

from ..events import EventBus

# Whisper hallucinates these on silence and near-silence — they are subtitle
# boilerplate from its training data, not anything the microphone heard. Observed
# in real sessions ("Thx for watching!" after a pause). Discarded outright: they
# are never a real utterance, and letting them through pollutes the log and risks
# a fuzzy wake match on noise.
_HALLUCINATION = re.compile(
    r"^\W*("
    r"th(an)?k?s? ?(you|for watching|for your attention)|"
    r"thank you( very much)?|thanks|thx( for watching)?|"
    r"bye|goodbye|see you( next time)?|"
    r"subscribe|like and subscribe|"
    r"subtitles? by.*|transcription by.*|amara\.org|"
    r"\[.*\]|\(.*\)|"
    r"you|uh|um|hmm|mm|ah|oh"
    r")\W*$", re.I)


def _collapse_repeats(text: str) -> str:
    """Cut a transcript short at the point it starts looping.

    Whisper's other failure mode on low-information audio is to repeat a phrase
    until it runs out of budget — often the initial_prompt, which is how HIVE
    once transcribed "open chrome. hive open chrome. hive open chrome." from an
    empty room and launched a browser. If a sentence repeats, keep the first and
    discard the rest; if the whole utterance is one repeated phrase, drop it.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if len(parts) < 2:
        return text
    out: list[str] = []
    seen: dict[str, int] = {}
    looped = False
    for p in parts:
        key = p.lower().strip(" .,!?")
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            looped = True
            break                       # a repeat means the model has started looping
        out.append(p)

    # Once a loop is detected in a multi-sentence utterance, void the whole
    # thing rather than keeping the prefix. People do not repeat a sentence
    # verbatim, so the transcript is untrustworthy end to end — and keeping the
    # prefix is what left "hive open chrome" intact and launched a browser.
    if looped and len(parts) >= 3:
        return ""
    return " ".join(out)


class Recognizer:
    def __init__(self, cfg, bus: EventBus):
        self.cfg, self.bus = cfg, bus
        self._model = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.model_name = cfg.get("asr.model", "small.en")
        self.device = cfg.get("asr.device", "cuda")
        self.compute_type = cfg.get("asr.compute_type", "int8_float16")
        self.beam_size = int(cfg.get("asr.beam_size", 1))
        self.hotwords = cfg.get("asr.hotwords") or None
        # hotwords alone was not enough: real sessions still transcribed the name
        # as "Hi" and even "5". initial_prompt conditions the decoder on preceding
        # context, which biases the very first token far more strongly — and the
        # first token is exactly where the wake word lives.
        self.initial_prompt = cfg.get("asr.initial_prompt") or None
        # Confidence gates. Noise transcribed as fluent text is worse than silence,
        # because the fluent text can match an intent and execute something.
        self.no_speech_max = float(cfg.get("asr.no_speech_max", 0.60))
        self.min_logprob = float(cfg.get("asr.min_logprob", -1.0))

    # -- lifecycle ----------------------------------------------------------
    def preload(self) -> None:
        """Load and warm the model off the critical path."""
        threading.Thread(target=self._load, name="hive-asr-load", daemon=True).start()

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        try:
            model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.cfg.models / "whisper"),
            )
        except Exception as exc:
            self.bus.log(f"ASR cuda load failed ({exc}); falling back to CPU", "warn")
            model = WhisperModel(
                self.model_name, device="cpu", compute_type="int8",
                download_root=str(self.cfg.models / "whisper"),
            )
        # First inference builds kernels and allocates workspace; do it now, not
        # on the user's first sentence.
        model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1, language="en")
        self._model = model
        self._ready.set()
        self.bus.log(f"ASR ready in {(time.monotonic()-t0)*1000:.0f}ms", "audio")

    def wait_ready(self, timeout: float = 60.0) -> bool:
        return self._ready.wait(timeout)

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    # -- inference ----------------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> str:
        if not self._ready.wait(30.0) or self._model is None:
            return ""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language="en",
                beam_size=self.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                hotwords=self.hotwords,
                initial_prompt=self.initial_prompt,
                temperature=0.0,
            )
            kept = []
            for seg in segments:
                # Whisper invents fluent text from noise, and does it with high
                # confidence in the token probabilities — no_speech_prob is the
                # only reliable signal that there was nothing there to transcribe.
                if getattr(seg, "no_speech_prob", 0.0) > self.no_speech_max:
                    self.bus.log(
                        f"(dropped, no speech p={seg.no_speech_prob:.2f}) {seg.text.strip()[:48]}",
                        "heard")
                    continue
                if getattr(seg, "avg_logprob", 0.0) < self.min_logprob:
                    self.bus.log(f"(dropped, low confidence) {seg.text.strip()[:48]}", "heard")
                    continue
                kept.append(seg.text)
            text = " ".join(kept)

        text = _collapse_repeats(text.strip())
        if not text or _HALLUCINATION.match(text):
            if text:
                self.bus.log(f"(discarded hallucination) {text[:60]}", "heard")
            return ""
        return text
