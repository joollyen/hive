"""Microphone capture and spectral analysis.

Two consumers hang off one stream:
  1. the VAD/ASR path, which needs every sample in order and on time
  2. the visualizer, which needs a cheap 128-band spectrum ~30x a second

The PortAudio callback runs on a realtime-ish thread. Everything in it is O(n) on
a 512-sample block with no allocation in the steady state, no locks, and no I/O.
Overrunning here does not drop a frame quietly — it clicks, audibly.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from ..events import AUDIO_LEVEL, EventBus

# Voice lives here. Analysing the full 0-8kHz range wastes most of the columns on
# bands that barely move during speech.
FMIN, FMAX = 80.0, 7000.0


@dataclass
class AudioFrame:
    samples: np.ndarray   # float32, mono, [blocksize]
    index: int


class Spectrum:
    """Log-spaced magnitude bands, smoothed for display.

    Kept separate from the capture stream so the TTS playback path can drive the
    exact same analysis — that is what makes the visualizer respond to HIVE's own
    voice while it speaks, not just to the microphone.
    """

    def __init__(self, bins: int = 128, samplerate: int = 16000, window: int = 1024):
        self.bins = bins
        self.samplerate = samplerate
        self.window = window
        self._buf = np.zeros(window, dtype=np.float32)
        self._han = np.hanning(window).astype(np.float32)
        self._out = np.zeros(bins, dtype=np.float32)
        self._peak = 1e-4

        freqs = np.fft.rfftfreq(window, 1.0 / samplerate)
        edges = np.geomspace(FMIN, min(FMAX, samplerate / 2 - 1), bins + 1)
        # Precompute which FFT rows fold into which display band.
        self._slices: list[tuple[int, int]] = []
        for i in range(bins):
            lo = int(np.searchsorted(freqs, edges[i], side="left"))
            hi = int(np.searchsorted(freqs, edges[i + 1], side="right"))
            self._slices.append((lo, max(hi, lo + 1)))

    def push(self, block: np.ndarray) -> None:
        n = len(block)
        if n >= self.window:
            self._buf[:] = block[-self.window:]
        else:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = block

    def compute(self, attack: float = 0.55, release: float = 0.14) -> np.ndarray:
        mag = np.abs(np.fft.rfft(self._buf * self._han))
        raw = np.empty(self.bins, dtype=np.float32)
        for i, (lo, hi) in enumerate(self._slices):
            raw[i] = mag[lo:hi].max() if hi > lo else 0.0

        # Perceptual compression, then normalise against a slowly-decaying peak so
        # quiet speech still fills the display without a loud sound pinning it.
        raw = np.log1p(raw * 8.0)
        self._peak = max(self._peak * 0.995, float(raw.max()), 1e-4)
        raw /= self._peak

        # Asymmetric smoothing: snap up on transients, ease down. Symmetric
        # smoothing reads as mushy and makes speech look like a sine wave.
        rising = raw > self._out
        coef = np.where(rising, attack, release).astype(np.float32)
        self._out += (raw - self._out) * coef
        return np.clip(self._out, 0.0, 1.0)


class MicCapture:
    def __init__(self, cfg, bus: EventBus):
        self.cfg = cfg
        self.bus = bus
        self.samplerate = int(cfg.get("audio.input.samplerate", 16000))
        self.blocksize = int(cfg.get("audio.input.blocksize", 512))
        self.gain = float(cfg.get("audio.input.gain", 1.0))
        device = cfg.get("audio.input.device", "default")
        self.device = None if device in (None, "", "default") else device

        self.frames: queue.Queue[AudioFrame] = queue.Queue(maxsize=256)
        self._spectrum = Spectrum(int(cfg.get("ui.spectrum_bins", 128)), self.samplerate)
        self._stream: sd.InputStream | None = None
        self._index = 0
        # Two separate facts, and v5 split them apart because they used to be one.
        #
        #   _speaking   HIVE is talking right now
        #   _muted      frames are not being fed to the endpointer
        #
        # They were the same flag, which is why "HIVE, stop" could never work:
        # the microphone was switched off for the entire duration of the thing
        # the operator was trying to interrupt. With barge-in on, audio keeps
        # flowing while HIVE speaks and the endpointer handles it differently.
        self._speaking = threading.Event()
        self._muted = threading.Event()
        self.barge_in = bool(cfg.get("audio.input.barge_in", True))
        self._pub_every = 1
        self._pub_count = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
            latency="low",
        )
        self._stream.start()
        self.bus.log(f"mic open @{self.samplerate}Hz blk={self.blocksize}", "audio")

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    # -- self-hearing guard -------------------------------------------------
    def mute(self) -> None:
        """Called when HIVE starts speaking.

        HIVE hears its own voice through the speakers, and left alone it
        endpoints that as user speech, transcribes itself and answers itself —
        a loop that is immediate and unrecoverable on an open-mic desktop.
        Cutting the feed is the reliable defence, and it is what this did.

        It also made "HIVE, stop" impossible: the one moment the operator needs
        to be heard is the one moment nothing was listening. So with barge-in
        enabled the feed stays open and the loop is broken further downstream —
        the endpointer slices this audio short and gates it on level, and the
        runtime only acts on it if it is a short command that HIVE did not just
        say itself. See Runtime._barge_in.
        """
        self._speaking.set()
        if not self.barge_in:
            self._muted.set()

    def unmute(self) -> None:
        with self.frames.mutex:              # drop whatever leaked in while muted
            self.frames.queue.clear()
        self._muted.clear()
        self._speaking.clear()

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    @property
    def is_speaking(self) -> bool:
        """Is HIVE talking? Frames captured now contain its voice as well."""
        return self._speaking.is_set()

    # -- hot path -----------------------------------------------------------
    def _callback(self, indata, nframes, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.bus.log(f"mic status {status}", "warn")
        block = indata[:, 0]
        if self.gain != 1.0:
            block = block * self.gain

        # Visualizer always sees the mic, even while muted, so the columns keep
        # breathing with the room instead of flatlining mid-conversation.
        self._spectrum.push(block)
        self._pub_count += 1
        if self._pub_count >= self._pub_every:
            self._pub_count = 0
            spec = self._spectrum.compute()
            rms = float(np.sqrt(np.mean(block * block)))
            self.bus.publish(
                AUDIO_LEVEL,
                rms=rms,
                peak=float(np.max(np.abs(block))),
                spectrum=spec.tolist(),
                source="mic",
            )

        if self._muted.is_set():
            return
        try:
            self.frames.put_nowait(AudioFrame(block.copy(), self._index))
            self._index += 1
        except queue.Full:
            pass   # consumer stalled; dropping the oldest audio beats blocking here
