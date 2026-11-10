# hive

A single-operator voice assistant for Linux. Speech recognition, speech synthesis, and the language models all run locally, and a deterministic safety layer sits in front of the model and can always overrule it.

> Demo recording: not yet recorded.

## Why I built it

I wanted a voice assistant I could trust on my own machine: nothing I say leaves the box, it never acts on a misheard command in a way that matters, and it answers fast enough to feel like a conversation. That meant building the whole speech pipeline locally on one consumer GPU, and putting rules, not the model, in charge of anything that touches the system.

## How it works

**Stack**

| Layer | Component |
|---|---|
| Speech recognition | faster-whisper `small.en` (CTranslate2, int8/float16) |
| Voice activity detection | Silero VAD v5 (ONNX) |
| Speech synthesis | Kokoro-82M (ONNX Runtime), a blended voice |
| Language models | Ollama, two tiers: `qwen3:1.7b` (fast, always loaded) and `qwen3:8b` (heavy, loaded on demand) |
| UI | PySide6 / Qt Quick (QML) with GLSL shader visuals |

The whole ML stack runs on CTranslate2 and ONNX Runtime, so there is no PyTorch install. That saves several GB of dependencies and seconds of startup time.

**Two-tier models from an 8 GB VRAM limit.** The target GPU (RTX 3070 Ti) has 8 GB. Whisper, Kokoro, and an 8B model cannot all stay resident with room to spare, so a small model stays loaded for short turns and the 8B model is brought in only when a turn needs it (long answers, analysis, memory consolidation) and released after a few minutes idle.

**Deterministic fast path.** Requests with one correct interpretation ("open Chrome", "what time is it", "make a note that...") are matched by an intent grammar and never reach a model. Fixed phrases are synthesized once and cached on disk, responses stream to the speaker one sentence at a time, and if a turn will be slow a cached filler line plays so the user never waits in silence.

**Objective / subjective split.** Every turn passes through two halves. The objective side (intent grammar, permission checks, deny lists, fact injection) is deterministic code. The subjective side is the language model, which phrases answers and handles open-ended conversation. When they disagree about an action, the objective side wins. The model can suggest; it cannot authorize.

**Memory.** Short-term memory is the current session. Long-term memory is a folder of Markdown notes (an Obsidian vault works), written when a session ends and searched by keyword during later sessions.

**HIVE-Q.** A second mode that takes a typed list of tasks and works through them one by one, narrating as it goes, and pauses whenever the operator speaks.

## Latency

Target: first audio within 800 ms of the operator finishing speaking. Measured on the development machine (i7-10700K, RTX 3070 Ti):

| Turn type | Time to first audio |
|---|---|
| Presence check, time, app launch, note | 56–101 ms |
| Memory recall or conversation, models warm | about 350–550 ms |
| First heavy-model question of a session | 341 ms to a filler line, full answer at about 4.7 s |

These exclude the end-of-speech pause. HIVE waits for 3 s of silence before deciding the operator has finished, a deliberate setting that trades responsiveness for not cutting people off mid-thought. It is configurable in `config/hive.toml`.

## Security notes

- **Local models only.** Speech, transcripts, prompts, and memory stay on the machine. No prompt or transcript is sent to any service.
- **Read-only web access (added in v6).** HIVE can run a web search and fetch a page to read it. Requests are GET only, with no cookies, logins, or forms, and with hard timeout, size, and redirect limits. URLs are resolved before connecting and anything resolving to a loopback, private, link-local, or reserved address is refused, so a URL from a search result cannot be pointed at the local Ollama server or other services on the network. What leaves the machine is a search query or a page request, never the conversation. Without a network connection HIVE runs fully offline and says so.
- **Session-scoped, no background listener.** The microphone is live only while the HIVE window is open. Closing it ends the session. There is no daemon.
- **Folder access only from typed input.** Granting HIVE access to a folder can only be done by typing in its terminal pane, never by voice, so a misheard path cannot reach a file operation. Credential and system locations (`~/.ssh`, `~/.gnupg`, `~/.config`, keyrings, `/etc`, `/proc`, `/dev`, and others) can never be granted.
- **Permission tiers.** Launching apps and URLs and writing inside its own memory folder are allowed. System queries, shell execution, and writes elsewhere require confirmation. Destructive commands (recursive deletes, `mkfs`, raw `dd` to devices, piping downloads into a shell, fork bombs, `sudo`, reading `/etc/shadow`, and similar) are denied outright, and no confirmation unlocks them. The deny list is matched broadly on purpose: a false positive costs one refusal, a false negative could cost a filesystem.
- **Whisper hallucinations.** Whisper can produce fluent text from background noise. Because intents are matched on that text, the wake-word requirement and the deny tier are what keep a hallucinated command from doing damage.

## Run it

Requirements: Linux, an NVIDIA GPU with recent drivers (8 GB VRAM recommended), a microphone, and [Ollama](https://ollama.com).

```bash
git clone https://github.com/joollyen/hive.git
cd hive
./install.sh            # userspace only, no sudo; creates .venv and downloads models
ollama pull qwen3:1.7b
ollama pull qwen3:8b
# set memory.vault_path in config/hive.toml to a folder of Markdown notes
./run.sh
```

Then say "HIVE, what time is it?"

Diagnostics:

```bash
./run.sh --selftest     # checks every layer and reports timings
./run.sh --miccheck     # confirms speech is reaching the voice activity detector
./run.sh --dryrun       # full turns from synthesized speech, no microphone (writes to the vault)
./run.sh --query        # HIVE-Q, the typed task-list mode
```

Optional per-domain answer profiles are described in `config/calibrations.example.yaml`.

## Known limitations

- **Tuned for one machine and one person.** Wake-word variants, voice activity thresholds, and app aliases were set from my own speech and setup. Expect to retune `config/hive.toml` and `config/apps.yaml`.
- **NVIDIA and Linux only** in practice. CUDA library handling assumes a specific driver and cuDNN layout, and the install has two known dependency conflicts that `install.sh` works around.
- **Slow end of turn.** The 3 s silence threshold makes every request feel slower than the pipeline actually is.
- **Wayland blocks window control.** HIVE can open applications, URLs, and files, but cannot focus, move, or resize existing windows.
- **One turn at a time.** Speech that arrives mid-turn is dropped, not queued. A stalled task in HIVE-Q can be abandoned but not killed, because Python cannot stop a running thread.
- **Memory recall is keyword search, not embeddings.** Fast and predictable, but it misses notes that use different words for the same thing.
- **The web reader only sees what an anonymous browser sees.** No logins, no JavaScript-rendered pages, and many large sites refuse automated clients.
- **No automated test suite in CI.** `--selftest` is a manual diagnostic, not a test suite.
- **The language model can still be wrong.** The safety layer constrains what it can do, not what it says.

## My role

This was built with AI-assisted development. I came up with the concept and the requirements, set the security rules (local-only, typed-only grants, the deny tier), tested every version by using it daily on my own machine, identified what failed, and decided what changed in each version.

## Next

Planned, not built:

- Automated tests for the intent grammar and the deny list, run in CI
- Embedding-based memory recall alongside keyword search
- Lower end-of-turn delay without cutting the operator off

## License

MIT. See [LICENSE](LICENSE).
