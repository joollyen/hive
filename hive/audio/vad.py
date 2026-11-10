"""Voice activity detection and utterance endpointing (Silero v5, ONNX, CPU).

Endpointing owns the largest single slice of perceived latency. The user stops
speaking; nothing can happen until we decide they have actually stopped. Every
millisecond of hangover here is a millisecond before HIVE answers.

Two mistakes are possible and they are not symmetric:
  - too short: HIVE interrupts mid-sentence, which is infuriating
  - too long : HIVE feels sluggish, which is merely disappointing

So the hangover is adaptive rather than fixed. An utterance ending on a clear
falling contour ("...open chrome") endpoints fast; one ending mid-phrase, or on a
filler like "uh", waits longer. Silero runs on CPU on purpose — it is ~1ms per
32ms frame there, and keeping it off the GPU leaves CUDA free for the two models
that genuinely need it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import onnxruntime as ort

from ..events import EventBus, State

CHUNK = 512          # Silero v5 is trained on exactly this at 16kHz
SR = 16000
CONTEXT = 64         # samples of the previous chunk the model expects prepended


@dataclass
class Utterance:
    audio: np.ndarray      # float32 mono @16k, including pre/post padding
    duration: float
    started_at: float      # monotonic, when speech began
    ended_at: float        # monotonic, when the endpointer fired
    # Captured while HIVE was talking, so it contains HIVE's own voice as well
    # as anything the operator said over it. Handled by a different path.
    during_speech: bool = False


class SileroVAD:
    """Silero v5 speech probability, one 512-sample frame at a time.

    The v5 ONNX graph expects each call to receive 64 samples of context carried
    over from the previous chunk, concatenated in front of the 512 new ones — 576
    samples total at 16kHz. This is not optional and it is not documented in the
    graph: the input is declared `[None, None]`, so feeding a bare 512 raises no
    error at all. It simply returns a speech probability of roughly 0.002 for
    clearly audible speech, and the endpointer never fires.
    """

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1     # single frame at a time; threads only add overhead
        opts.log_severity_level = 4
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._sr = np.array(SR, dtype=np.int64)

    def reset(self) -> None:
        self._state[:] = 0.0
        self._context[:] = 0.0

    def __call__(self, chunk: np.ndarray) -> float:
        if len(chunk) != CHUNK:
            buf = np.zeros(CHUNK, dtype=np.float32)
            buf[: min(len(chunk), CHUNK)] = chunk[:CHUNK]
            chunk = buf
        frame = np.concatenate(
            [self._context, chunk.reshape(1, -1).astype(np.float32)], axis=1
        )
        out, self._state = self.sess.run(
            ["output", "stateN"],
            {"input": frame, "state": self._state, "sr": self._sr},
        )
        self._context = frame[:, -CONTEXT:]
        return float(out[0][0])


class Endpointer(threading.Thread):
    """Turns a stream of frames into discrete utterances."""

    daemon = True

    def __init__(self, cfg, bus: EventBus, capture, on_utterance: Callable[[Utterance], None]):
        super().__init__(name="hive-vad")
        self.cfg, self.bus, self.capture = cfg, bus, capture
        self.on_utterance = on_utterance
        self.vad = SileroVAD(str(cfg.models / "vad" / "silero_vad.onnx"))

        self.threshold = float(cfg.get("vad.threshold", 0.55))
        self.min_speech = float(cfg.get("vad.min_speech_ms", 180)) / 1000.0
        self.min_silence = float(cfg.get("vad.min_silence_ms", 380)) / 1000.0
        self.pad = float(cfg.get("vad.speech_pad_ms", 180)) / 1000.0
        self.max_utt = float(cfg.get("vad.max_utterance_s", 30.0))

        # Barge-in. While HIVE is speaking the microphone is hearing HIVE, so
        # this path exists to find the one thing worth acting on in that audio:
        # the operator talking over the top of it.
        #
        # Two adjustments, both necessary. HIVE's own speech has no gap longer
        # than the sentence pause, so under the normal hangover it never
        # endpoints at all — the whole answer becomes one utterance that arrives
        # thirty seconds too late to interrupt anything. So it is sliced on a
        # fixed short window instead. And every slice would otherwise be sent to
        # the recogniser, which is a transcription every second and a half for
        # the length of the answer, competing with the synthesiser for the same
        # GPU — so a slice has to be measurably louder than HIVE's own voice
        # before it is worth looking at.
        self.barge_window = float(cfg.get("vad.barge_in_window_s", 2.5))
        self.barge_ratio = float(cfg.get("vad.barge_in_ratio", 1.9))
        self._echo_rms = 0.0            # running level of HIVE-through-the-speakers
        self._barge: list[np.ndarray] = []
        self._barge_pre: list[np.ndarray] = []
        self._barge_loud = 0.0
        self._barge_quiet = 0.0
        self._echo_frames = 0

        self._stop = threading.Event()
        self._frame_s = CHUNK / SR
        self._preroll: list[np.ndarray] = []
        self._preroll_max = max(1, int(self.pad / self._frame_s))

    def stop(self) -> None:
        self._stop.set()

    # -- barge-in -----------------------------------------------------------
    def _note_echo(self, chunk: np.ndarray) -> None:
        """Track how loud HIVE's own voice reads at the microphone.

        Slow on purpose. The operator talking over HIVE is part of this signal
        too, so a fast average would climb to meet them and the gate would close
        exactly when it is needed. At this coefficient a second of speech moves
        the floor by a few percent.
        """
        rms = self._rms(chunk)
        self._echo_frames += 1
        # Converge quickly for the first half second, then slowly. A slow
        # average from a standing start reads far too low while it climbs, and
        # every ordinary syllable of HIVE's own speech clears the gate during
        # that window — which is a transcription each, for nothing.
        alpha = 0.25 if self._echo_frames < 16 else 0.015
        self._echo_rms = ((1.0 - alpha) * self._echo_rms + alpha * rms
                          if self._echo_rms else rms)

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))

    def _barge_reset(self) -> None:
        self._echo_rms = 0.0
        self._echo_frames = 0
        self._barge.clear()
        self._barge_pre.clear()
        self._barge_loud = self._barge_quiet = 0.0

    def _barge_step(self, chunk: np.ndarray) -> np.ndarray | None:
        """One frame of the interrupt detector. Returns audio when it has some.

        Triggered by the *rise*, not by a clock. An earlier version cut fixed
        windows out of the incoming audio and looked at each one, which mostly
        produced slices straddling the moment the operator started talking —
        half HIVE, half Jay, and the recogniser reported whichever had more of
        it. Starting the capture where the level jumps puts the interruption at
        the front of the clip, which is the only place the anchored grammar can
        see it.
        """
        # Not armed until the floor means something. Sixteen frames is half a
        # second of HIVE talking, which is enough to know how loud it reads.
        armed = self._echo_rms > 0.0 and self._echo_frames >= 16
        loud = armed and self._rms(chunk) >= self._echo_rms * self.barge_ratio

        if not self._barge:
            self._barge_pre.append(chunk)
            if len(self._barge_pre) > 8:          # ~256ms of run-up
                self._barge_pre.pop(0)
            self._barge_loud = self._barge_loud + self._frame_s if loud else 0.0
            if self._barge_loud >= 0.16:
                self._barge = list(self._barge_pre)
                self._barge_pre.clear()
                self._barge_quiet = 0.0
            return None

        self._barge.append(chunk)
        self._barge_quiet = 0.0 if loud else self._barge_quiet + self._frame_s
        held = len(self._barge) * self._frame_s
        if self._barge_quiet >= 0.42 or held >= self.barge_window:
            audio = np.concatenate(self._barge)
            self._barge.clear()
            self._barge_loud = self._barge_quiet = 0.0
            return audio
        return None

    # -- adaptive hangover --------------------------------------------------
    def _hangover_for(self, probs: list[float], voiced_s: float) -> float:
        """How long to wait before declaring the utterance over.

        The obvious heuristic — short utterance, endpoint fast — is wrong here,
        and observably so. People say the name, pause, then say the command:
        "HIVE, ... are you there?". Under an aggressive hangover that lone "HIVE"
        endpoints during the comma and arrives as its own utterance, leaving the
        command stranded in a second fragment with no wake word. Both are then
        discarded and HIVE appears deaf.

        So the very shortest utterances — the ones most likely to be just a name —
        get the *longest* wait, not the shortest. Only a mid-length utterance with
        a confidently-falling tail is endpointed early.

        The multipliers are deliberately gentle now that the base is measured in
        seconds rather than hundreds of milliseconds. They were 1.8 and 0.8,
        which is fine either side of 380ms and turns a three-second wait into
        five and a half.
        """
        base = self.min_silence
        if voiced_s < 0.95:
            return base * 1.15      # probably just the wake word; the command follows
        if voiced_s < 1.6 and len(probs) >= 6:
            # Confident, monotonically-falling tail => speaker is done.
            if max(probs[-3:]) < self.threshold * 0.45:
                return base * 0.75
        if voiced_s > 4.0:
            return base * 1.15      # long thought, likely mid-sentence pause
        return base

    def run(self) -> None:  # noqa: C901 - a state machine reads better in one piece
        in_speech = False
        voiced: list[np.ndarray] = []
        probs: list[float] = []
        silence_run = 0.0
        speech_run = 0.0
        started_at = 0.0

        while not self._stop.is_set():
            try:
                frame = self.capture.frames.get(timeout=0.25)
            except Exception:
                continue

            block = frame.samples
            # Frames arrive at blocksize; Silero wants exactly CHUNK.
            for off in range(0, len(block), CHUNK):
                chunk = block[off: off + CHUNK]
                if len(chunk) < CHUNK:
                    chunk = np.pad(chunk, (0, CHUNK - len(chunk)))

                # --- HIVE is the one talking -----------------------------
                # Everything arriving now contains its voice, so the ordinary
                # state machine below has nothing to do: it is built to find
                # utterances between silences, and there is no silence. The only
                # question worth asking of this audio is whether someone is
                # talking over the top of it.
                if getattr(self.capture, "is_speaking", False):
                    if not self._barge:
                        self._note_echo(chunk)   # frozen once a candidate starts,
                                                 # or the floor climbs to meet it
                    found = self._barge_step(chunk)
                    if found is not None:
                        self.on_utterance(Utterance(
                            found, len(found) / SR, time.monotonic(),
                            time.monotonic(), during_speech=True))
                    if in_speech:
                        # Jay was mid-sentence when HIVE started. Keep what he
                        # said rather than discarding it.
                        in_speech = False
                        voiced, probs = [], []
                    continue
                if self._echo_rms or self._barge or self._barge_pre:
                    self._barge_reset()          # HIVE stopped; back to normal
                    self.vad.reset()

                p = self.vad(chunk)
                speaking = p >= self.threshold

                if not in_speech:
                    self._preroll.append(chunk)
                    if len(self._preroll) > self._preroll_max:
                        self._preroll.pop(0)
                    if speaking:
                        speech_run += self._frame_s
                        if speech_run >= self.min_speech:
                            in_speech = True
                            started_at = time.monotonic()
                            voiced = list(self._preroll)     # keep the onset
                            probs = [p]
                            silence_run = 0.0
                            self._preroll.clear()
                            self.bus.set_state(State.LISTENING)
                    else:
                        speech_run = 0.0
                    continue

                # --- in speech ---
                voiced.append(chunk)
                probs.append(p)
                if speaking:
                    silence_run = 0.0
                else:
                    silence_run += self._frame_s

                voiced_s = len(voiced) * self._frame_s
                over_time = voiced_s >= self.max_utt
                if over_time or silence_run >= self._hangover_for(probs, voiced_s):
                    audio = np.concatenate(voiced) if voiced else np.zeros(0, np.float32)
                    # Trim most of the trailing silence but leave `pad` so the final
                    # consonant is not clipped — Whisper drops word-final stops badly.
                    keep = int(max(0.0, silence_run - self.pad) * SR)
                    if keep > 0:
                        audio = audio[:-keep] if keep < len(audio) else audio
                    in_speech = False
                    speech_run = 0.0
                    self._preroll.clear()
                    self.vad.reset()

                    dur = len(audio) / SR
                    if dur >= self.min_speech:
                        self.on_utterance(
                            Utterance(audio, dur, started_at, time.monotonic())
                        )
                    voiced, probs = [], []
