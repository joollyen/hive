"""Two-tier local language model client (Ollama).

Everything runs on this machine. No prompt, no transcript and no fragment of the
vault ever leaves it.

The two tiers exist because of an 8GB VRAM budget shared with the desktop:

  fast (qwen3:1.7b, ~1.4GB)  resident for the whole session. Handles conversation,
                             short answers and anything the deterministic grammar
                             did not catch but that does not need real reasoning.
  heavy (qwen3:8b, ~5.2GB)   loaded on demand, released after 90 seconds. This is
                             the model that knows the eCPPT follows the eJPT.

Measured on this box: first token 171ms fast, 186ms heavy — escalation costs
almost nothing in time-to-first-audio, it costs VRAM. So the escalation heuristic
optimises for "is this question worth 5GB", not "is this question slow".

Qwen3's reasoning traces are disabled outright. They add seconds before the first
user-visible token, which for a voice interface is not a tradeoff, it is a defect.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Iterator

import httpx

from ..events import EventBus


@dataclass
class Turn:
    role: str
    content: str


class LLM:
    def __init__(self, cfg, bus: EventBus):
        self.cfg, self.bus = cfg, bus
        self.host = cfg.get("llm.host", "http://127.0.0.1:11434").rstrip("/")
        self.fast = cfg.get("llm.fast_model", "qwen3:1.7b")
        self.heavy = cfg.get("llm.heavy_model", "qwen3:8b")
        self.fast_keep = cfg.get("llm.fast_keep_alive", "30m")
        self.heavy_keep = cfg.get("llm.heavy_keep_alive", "90s")
        self.think = bool(cfg.get("llm.think", False))
        self.temperature = float(cfg.get("llm.temperature", 0.55))
        self._client = httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))
        self._escalate_words = int(cfg.get("llm.escalate_on_words", 28))
        self._escalate_kw = [k.lower() for k in cfg.get("llm.escalate_keywords", [])]
        self._lock = threading.Lock()

    # -- availability -------------------------------------------------------
    def available(self) -> bool:
        try:
            r = self._client.get(f"{self.host}/api/version", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def preload(self) -> None:
        """Warm the fast tier so the first real turn does not pay load time."""
        def _warm() -> None:
            try:
                self._client.post(
                    f"{self.host}/api/chat",
                    json={"model": self.fast, "messages": [{"role": "user", "content": "hi"}],
                          "stream": False, "keep_alive": self.fast_keep, "think": False,
                          "options": {"num_predict": 1}},
                    timeout=90.0,
                )
                self.bus.log(f"LLM warm: {self.fast}", "llm")
            except Exception as exc:
                self.bus.log(f"LLM warm failed: {exc}", "warn")
        threading.Thread(target=_warm, name="hive-llm-warm", daemon=True).start()

    # -- escalation ---------------------------------------------------------
    def should_escalate(self, text: str, category: str = "general") -> bool:
        """Is this worth spending 5GB of VRAM on?"""
        t = text.lower()
        if any(k in t for k in self._escalate_kw):
            return True
        if len(t.split()) >= self._escalate_words:
            return True
        if category in ("analysis", "synthesis", "writing", "research"):
            return True
        return False

    def release_heavy(self) -> None:
        """Drop the heavy model now rather than waiting out its keep-alive.

        Called before anything else needs the GPU — the difference between 6.5GB
        and 1.4GB resident decides whether the next CUDA allocation succeeds.
        """
        try:
            self._client.post(
                f"{self.host}/api/generate",
                json={"model": self.heavy, "keep_alive": 0, "prompt": ""}, timeout=10.0,
            )
        except Exception:
            pass

    # -- generation ---------------------------------------------------------
    def stream(self, messages: list[dict], heavy: bool = False,
               num_predict: int | None = None) -> Iterator[str]:
        """Yield content deltas. Feeds straight into Speaker.say_stream."""
        model = self.heavy if heavy else self.fast
        keep = self.heavy_keep if heavy else self.fast_keep
        if num_predict is None:
            num_predict = int(self.cfg.get(
                "llm.heavy_num_predict" if heavy else "llm.fast_num_predict", 220))

        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
                "top_p": 0.9,
                "repeat_penalty": 1.08,
                # Stop the model from narrating structure it should not be
                # speaking in the first place.
                "stop": ["\n\n\n"],
            },
        }
        try:
            with self._lock, self._client.stream(
                "POST", f"{self.host}/api/chat", json=body
            ) as resp:
                if resp.status_code != 200:
                    self.bus.log(f"LLM http {resp.status_code}", "error")
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except Exception as exc:
            self.bus.log(f"LLM stream failed: {exc}", "error")

    def complete(self, messages: list[dict], heavy: bool = False,
                 num_predict: int | None = None) -> str:
        return "".join(self.stream(messages, heavy=heavy, num_predict=num_predict)).strip()


# --- output hygiene --------------------------------------------------------
# Small models drift into chat-assistant habits that are fine on screen and awful
# out loud. Cheaper and far more reliable to strip them than to prompt against them.

_STRIP = [
    (re.compile(r"<think>.*?</think>", re.S), ""),          # stray reasoning traces
    (re.compile(r"^\s*(Sure|Certainly|Absolutely|Of course|Great question)[!,.]?\s*",
                re.I), ""),
    (re.compile(r"^\s*(I'd be happy to|I would be happy to)[^.]*\.\s*", re.I), ""),
    (re.compile(r"\bAs an AI( language model)?\b[^.]*\.\s*", re.I), ""),
    # Hollow closers. The persona forbids these and the curiosity layer asks
    # again for the turns it is armed on, and qwen3:1.7b writes them anyway.
    #
    # This runs after the answer has been streamed, so it cleans what goes into
    # short-term memory and the vault rather than what comes out of the speakers
    # — buffering a sentence to catch it before playback would cost exactly the
    # first-audio latency the whole streaming path exists to protect.
    (re.compile(r"\s*(Let me know|Just let me know|Feel free to (ask|let me know))"
                r"\b[^.!?]*[.!?]\s*$", re.I), ""),
    (re.compile(r"\s*(Would you like me to|Do you want me to|Shall I)"
                r"\b[^.!?]*\?\s*$", re.I), ""),
    # A promise of an artefact. HIVE cannot create a file by saying it will, and
    # a sentence that says otherwise is the one thing in a transcript that is
    # simply false — "let me share a detailed summary when ready" was said in a
    # real session and nothing was ever shared. The persona forbids these and
    # qwen3:1.7b writes them anyway.
    (re.compile(r"\s*(?:I'?ll|I will|Let me|I'?m going to|I can)\s+"
                r"(?:go ahead and\s+)?"
                r"(?:creat\w*|writ\w*|put together|draft\w*|prepar\w*|shar\w*|"
                r"compil\w*|generat\w*|sav\w*|assembl\w*)\s+"
                r"[^.!?]*\b(?:file|note|document|summary|write ?up|report|"
                r"markdown|md|doc)\b[^.!?]*[.!?]", re.I), ""),
]


def clean_for_speech(text: str) -> str:
    """Tidy generated text for the transcript and the vault.

    Runs *after* the answer has been streamed, so this is not what reaches the
    speakers — it is what reaches memory. The layout stripping it used to do
    lives in audio/prosody.py now, and is shared with the synthesis path
    deliberately: the two used to disagree, so an enumerator was rewritten one
    way for the ear ("1." spoken as "one.") and deleted outright for the record.
    What the vault holds should be what was actually said.
    """
    from ..audio.prosody import despatialise, repunctuate

    for pat, rep in _STRIP:
        text = pat.sub(rep, text)
    text = repunctuate(despatialise(text))
    return re.sub(r"\s+", " ", text).strip()
