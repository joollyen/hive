"""Session orchestrator.

Owns the lifecycle: boot the engines, run the turn loop, consolidate to the vault
on close. HIVE is session-scoped by design — closing the window ends it, and
nothing survives except what gets written to Obsidian on the way out.

Threading model, and why it is threads rather than asyncio:

    main            Qt event loop, UI only
    hive-vad        endpointing, drains the capture queue
    hive-turn       one turn at a time: ASR, cognition, speech
    PortAudio x2    capture and playback callbacks
    load threads    model warm-up, phrase cache, transient timers

The heavy work — CTranslate2, ONNX Runtime, the HTTP stream to Ollama — all
releases the GIL, so threads give real parallelism here without asyncio's
colouring problem or the awkwardness of bridging a loop into Qt's.

Exactly one turn runs at a time. Utterances arriving mid-turn are dropped rather
than queued: if you interrupt HIVE, you meant to replace what you said, not to
stack another request behind it.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from datetime import datetime

from .audio.asr import Recognizer
from .audio.capture import MicCapture
from .audio.tts import Speaker
from .audio.vad import Endpointer, Utterance
from .clip import Clippings
from .cognition.calibration import Calibrator
from .cognition.callosum import Callosum
from .cognition.curiosity import Curiosity
from .cognition.intents import detect_wake, is_interrupt
from .cognition.llm import LLM, clean_for_speech
from .cognition.objective import ObjectiveHemisphere
from .cognition.prefetch import SlowThinker
from .cognition.speculate import Speculator
from .cognition.subjective import SubjectiveHemisphere
from .events import (ENGINE_READY, LOG_LINE, SESSION_SAVED, SPEAK_CHUNK,
                     TRANSCRIPT_FINAL, EventBus, State, Stopwatch)
from .memory.associations import Associations
from .memory.ltm import LongTermMemory, slugify_title
from .memory.stm import ShortTermMemory
from .net import Web
from .nightmode import NightMode
from .query.runner import QueryRunner
from .skills.launcher import Launcher
from .skills.workspace import Workspace
from .terminal import Terminal


class Runtime:
    def __init__(self, cfg, bus: EventBus):
        self.cfg, self.bus = cfg, bus

        self.stm = ShortTermMemory(cfg)
        self.ltm = LongTermMemory(cfg, bus)
        self.launcher = Launcher(cfg, bus)
        self.workspace = Workspace(cfg, bus, self.ltm)
        # Pasted material. Session-scoped like short-term memory, and separate
        # from it deliberately — a quoted source is not a turn. See clip.py.
        self.clippings = Clippings(cfg, bus)
        # Set by the UI once Qt owns a thread that may touch QClipboard.
        self.clipboard = None

        self.capture = MicCapture(cfg, bus)
        self.recognizer = Recognizer(cfg, bus)
        self.speaker = Speaker(cfg, bus)
        self.llm = LLM(cfg, bus)

        self.night = NightMode(cfg, bus, self)
        self.prefetch = SlowThinker(cfg, bus, self.llm, self.ltm, self.stm)
        self.associations = Associations(cfg, bus, self.ltm)
        # The network, and the queue that uses it most. The runner needs the
        # whole runtime — it takes the same turn lock a spoken utterance takes —
        # so it is constructed here and handed to the objective side afterwards.
        self.web = Web(cfg, bus)
        self.query = QueryRunner(cfg, bus, self)
        # The objective side gets the model and the index only for the document
        # writer: a requested write-up is prose bound for a file, and its recall
        # section is built from indexed mentions rather than from memory.
        self.objective = ObjectiveHemisphere(cfg, bus, self.launcher, self.ltm,
                                             self.speaker, self.workspace, self.night,
                                             llm=self.llm,
                                             associations=self.associations,
                                             web=self.web, query=self.query)
        self.subjective = SubjectiveHemisphere(cfg, bus, self.ltm)
        self.curiosity = Curiosity(cfg, bus, self.associations, self.subjective,
                                   self.night)
        self.calibrator = Calibrator(cfg, bus)
        self.speculate = Speculator(cfg, bus, self.speaker, self.llm,
                                    self.prefetch, self.calibrator)
        self.callosum = Callosum(cfg, bus, self.objective, self.subjective,
                                 self.llm, self.speaker, self.stm, self.ltm,
                                 self.curiosity, self.calibrator,
                                 clippings=self.clippings)
        # One list, two owners. The callosum appends as actions run; the document
        # writer reads it so a requested write-up records what was actually done
        # during the conversation rather than only what was said.
        self.objective.actions_log = self.callosum.actions_log
        self.terminal = Terminal(cfg, bus, self.workspace, self.speaker,
                                 on_request=self.handle_typed, night=self.night,
                                 associations=self.associations,
                                 calibrator=self.calibrator, query=self.query,
                                 web=self.web, stm=self.stm,
                                 clippings=self.clippings,
                                 save_fn=self.save_now)

        self.endpointer: Endpointer | None = None
        self._logfile = None
        self._turn_lock = threading.Lock()
        self._followup_until = 0.0
        self._followup_window = float(cfg.get("wake.followup_window_s", 15.0))
        self._followup_extends = bool(cfg.get("wake.followup_extends", True))
        self._running = threading.Event()
        self._closing = threading.Event()
        # Set once stop() has finished, consolidation included. The UI waits on
        # this instead of guessing at a timeout — see ui/app.py.
        self.stopped = threading.Event()
        self._budget_ms = int(cfg.get("latency.target_first_audio_ms", 800))
        self._warn_budget = bool(cfg.get("latency.warn_over_budget", True))
        # Barge-in state. The deque is what HIVE has said in the last few
        # seconds, kept so it can tell its own voice from Jay's.
        self._spoken_recently: deque[str] = deque(maxlen=8)
        self._last_barge_check = 0.0
        self._barge_min_gap = float(cfg.get("vad.barge_in_min_gap_s", 0.7))
        # Where the session ended up, read back off disk rather than assumed.
        # Printed on the way out so the answer to "did it save" is on screen
        # instead of in the vault folder.
        self.last_note = ""
        self.vault_receipt = ""

    # -- lifecycle ----------------------------------------------------------
    def attach_clipboard(self, clipboard) -> None:
        """Hand the engine the system clipboard, once Qt exists to own it.

        Late rather than in the constructor because QClipboard needs a
        QGuiApplication, and the runtime is built to be constructible without
        one — the self test and the dry run both do it.
        """
        self.clipboard = clipboard
        self.terminal.clipboard = clipboard

    def _open_log(self) -> None:
        """Mirror the event log to a file.

        Launched from the app grid there is no terminal, so stderr goes nowhere
        and a failure looks like HIVE simply not responding. This is the only
        record of what happened in a real session.
        """
        try:
            d = self.cfg.root / ".logs"
            d.mkdir(parents=True, exist_ok=True)
            self._logfile = (d / "hive.log").open("a", encoding="utf-8", buffering=1)
            self._logfile.write(f"\n===== session {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")

            def sink(evt) -> None:
                if self._logfile is None:
                    return
                try:
                    self._logfile.write(
                        f"{datetime.now():%H:%M:%S} [{evt.payload.get('kind','info')}] "
                        f"{evt.payload.get('text','')}\n")
                except (ValueError, OSError):
                    pass

            self.bus.subscribe(LOG_LINE, sink)
        except OSError:
            self._logfile = None

    def start(self) -> None:
        self._open_log()
        self.bus.log("HIVE booting", "boot")
        self.ltm.ensure()
        self.subjective.load()
        self.workspace.load()
        self.night.start_watching()
        # Indexing the vault is pure I/O over a few hundred small files. It runs
        # in the background so boot is not held up, and connections simply do not
        # fire for the first second or two of a session.
        self.associations.build_async()

        if self.subjective.model.voice:
            self.speaker.voice = self.subjective.model.voice
        self.speaker.speed = self.subjective.model.speech_rate
        # Which applications he actually opens, so the phrase cache can render
        # "Opening Spotify" ahead of the first time he asks for it this session.
        self.cfg.data["_app_counts"] = dict(self.subjective.model.app_counts)

        # Tell the microphone when HIVE is talking. Without this HIVE hears
        # itself through the speakers and answers itself, forever — with
        # barge-in off that means cutting the feed outright, and with it on it
        # means handling that audio as a possible interruption instead.
        self.speaker.set_on_speaking_change(
            lambda speaking: self.capture.mute() if speaking else self.capture.unmute()
        )
        # What HIVE is saying, as it says it — the reference the barge-in path
        # checks a transcript against to decide whether it just heard itself.
        self.bus.subscribe(SPEAK_CHUNK, self._note_spoken)

        self.recognizer.preload()
        self.speaker.preload(on_ready=self._greet)
        if self.llm.available():
            self.llm.preload()
        else:
            self.bus.log("Ollama not reachable — conversation disabled", "warn")

        # Settle the online/offline question in the background. It is one HEAD
        # request, but it gates what HIVE tells the model about itself on the
        # very first turn, and paying for it on that turn would put a DNS round
        # trip inside the latency budget.
        threading.Thread(
            target=lambda: self.bus.log(
                f"network {'reachable' if self.web.online(force=True) else 'unreachable'}",
                "net"),
            name="hive-net-probe", daemon=True).start()

        # Restore whatever was left in the HIVE-Q editor last session. The query
        # is a written document; closing HIVE should no more discard it than
        # closing a text editor discards the file.
        try:
            self.query.buffer = self.ltm.load_state("query", {}).get("buffer", "")
        except Exception:
            pass

        self.speculate.start()
        self.capture.start()
        self.endpointer = Endpointer(self.cfg, self.bus, self.capture, self._on_utterance)
        self.endpointer.start()
        self._running.set()
        self.bus.set_state(State.DORMANT)

    def _greet(self) -> None:
        """Spoken once TTS is warm, so the first thing heard is not a stutter."""
        # Lift the boot veil before speaking, not after: the greeting takes about
        # a second and a half, and the user should be watching the visualizer
        # while they hear it rather than staring at a placeholder.
        self.bus.publish(ENGINE_READY, tts=self.speaker.provider,
                         asr=self.recognizer.ready)
        bank = "greeting.returning" if self.subjective.returning else "greeting.cold"
        self.speaker.say_line(bank)
        self.bus.set_state(State.DORMANT)

    def stop(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        self._running.clear()
        self.bus.set_state(State.CLOSING)

        # The very first thing, ahead of every teardown step below, because
        # every one of them can block: the query unwinds a running task, a
        # write-up in flight is waited on for minutes, and the summariser takes
        # up to a minute by itself. All of that used to stand between "poweroff"
        # and the conversation reaching disk, and anything that killed the
        # process in the meantime took the session with it.
        #
        # Nothing here needs a model, a network or a lock. It is the data that
        # has been sitting in memory all session, written out. From this line
        # onward the session is safe.
        try:
            first = self.save_transcript()
            if first:
                self.bus.log(f"transcript safe in {first}", "memory")
        except Exception as exc:
            self.bus.log(f"early transcript save failed: {exc}", "error")

        try:
            # The query first: it holds the turn lock while a task runs, and
            # every teardown step below wants that lock free. Its own stop()
            # ends the run and waits briefly rather than indefinitely — a task
            # that will not unwind must not be able to prevent the session
            # being written to the vault.
            self.query.stop()
            self.speculate.stop()
            self.night.stop()
            if self.endpointer:
                self.endpointer.stop()
            self.capture.stop()
        except Exception:
            pass
        # A requested write-up runs on its own thread and outlives its turn.
        # Quitting before it lands would lose the file — which is the failure
        # this whole path exists to remove, arriving by a different door.
        try:
            self.objective.wait_for_writes()
        except Exception as exc:
            self.bus.log(f"write-up wait failed: {exc}", "warn")
        # The query survives the session. Saved unconditionally, before
        # consolidation, so an unfinished list is still there tomorrow even if
        # the summariser falls over on the way out.
        try:
            self.ltm.save_state("query", {"buffer": self.query.buffer})
        except Exception as exc:
            self.bus.log(f"query buffer save failed: {exc}", "warn")
        try:
            self.consolidate()
        finally:
            try:
                self.speaker.close()
            except Exception:
                pass
            try:
                self.web.close()
            except Exception:
                pass
        # Say where it went, on the way out. The whole complaint behind this
        # path is that the only way to find out whether HIVE had saved anything
        # was to go and look in Obsidian.
        if self.vault_receipt:
            self.bus.log(f"session saved: {self.vault_receipt}", "memory")
        elif self.stm.user_turns:
            self.bus.log("SESSION NOT SAVED — nothing was verified on disk",
                         "error")
        self.bus.log("HIVE offline", "boot")
        self.stopped.set()
        if self._logfile is not None:
            try:
                self._logfile.flush()
                self._logfile.close()
            finally:
                self._logfile = None

    # -- turn loop ----------------------------------------------------------
    def _on_utterance(self, utt: Utterance) -> None:
        if not self._running.is_set():
            return
        # Spoken over the top of HIVE. Never a new request — a turn is already
        # running and holding the lock — so it goes to the interrupt path, which
        # deliberately does not take the lock. Queuing it behind the turn it is
        # trying to stop would be the one arrangement guaranteed not to work.
        if utt.during_speech:
            threading.Thread(target=self._barge_in, args=(utt,),
                             name="hive-bargein", daemon=True).start()
            return
        # Non-blocking: if a turn is in flight, this utterance is discarded.
        #
        # Unless the turn in flight belongs to HIVE-Q, in which case it is not a
        # turn at all — it is background work, and the person in the room
        # outranks it. So the queue is asked to hold at the next task boundary
        # and this utterance waits for the lock rather than being thrown away.
        # Dropping it would mean speaking to HIVE during a query silently did
        # nothing, which is the worst of the available behaviours.
        if not self._turn_lock.acquire(blocking=False):
            if not self._yield_query("something you said"):
                return
        threading.Thread(target=self._run_turn, args=(utt,),
                         name="hive-turn", daemon=True).start()

    def _yield_query(self, why: str) -> bool:
        """Take the floor back from a running query. True if the lock is held."""
        if self.query is None or not self.query.running:
            return False
        self.query.request_pause(why)
        wait = float(self.cfg.get("query.lock_wait_s", 120.0))
        if self._turn_lock.acquire(timeout=wait):
            return True
        self.query.release_pause()
        return False

    # -- barge-in ------------------------------------------------------------
    def _note_spoken(self, evt) -> None:
        """Remember what HIVE just said, so it can recognise itself later."""
        text = (evt.payload.get("text") or "").strip().lower()
        if text:
            self._spoken_recently.append(text)

    def _is_own_voice(self, heard: str) -> bool:
        """Did the microphone just pick up HIVE rather than Jay?

        Checked only after the interrupt grammar has already matched, and that
        order matters. The transcript of someone talking over an assistant
        contains both of them, so testing for overlap first would throw away
        every real barge-in — the assistant's words are in there by definition.
        Testing it second means this only has to answer a much narrower
        question: HIVE said a stop word itself, and is about to obey it.
        """
        from rapidfuzz import fuzz

        low = heard.strip().lower()
        if not low:
            return True
        return any(fuzz.partial_ratio(low, said) >= 88
                   for said in list(self._spoken_recently))

    def _barge_in(self, utt: Utterance) -> None:
        """Stop talking, if that is what was just said.

        Everything about this path is narrow on purpose. It runs on audio that
        contains HIVE's own voice, so it can only afford to act on the one
        instruction that is unambiguous and whose cost of being wrong is a
        sentence cut short rather than an action taken.
        """
        now = time.monotonic()
        if now - self._last_barge_check < self._barge_min_gap:
            return                      # one look per gap; the rest is HIVE's voice
        self._last_barge_check = now
        if not self.speaker.speaking:
            return                      # it stopped on its own between then and now

        text = self.recognizer.transcribe(utt.audio)
        if not text:
            return
        _addressed, rest = detect_wake(text, self.cfg)
        if not is_interrupt(rest or text):
            self.bus.log(f"(over me) {text[:56]}", "heard")
            return
        if self._is_own_voice(text):
            self.bus.log(f"(that was me) {text[:56]}", "heard")
            return

        self.speaker.interrupt()
        self.stm.pending_confirm = None
        self.bus.log(f"stopped: {text[:56]}", "action")
        # Leave the follow-up window open. Someone who has just talked over an
        # answer is about to correct it, and making them say the name again to
        # do that is the opposite of stopping when told.
        self._followup_until = time.monotonic() + self._followup_window
        self.bus.set_state(State.LISTENING)

    def handle_typed(self, text: str) -> str:
        """Run a turn that arrived from the terminal rather than the microphone.

        No wake word and no ASR — the text is exact. Everything downstream is
        identical, so a typed request and a spoken one share memory, permissions
        and the same spoken reply.
        """
        if not text.strip():
            return ""
        if not self._turn_lock.acquire(timeout=20.0):
            # Same rule as the microphone: a typed request outranks the queue.
            if not self._yield_query("something you typed"):
                return "busy with something else"
        sw = Stopwatch(self.bus, "typed")
        try:
            self.bus.log(f"> {text[:70]}  (typed)", "user")
            outcome = self.callosum.handle(text, sw)
            if outcome.shutdown:
                threading.Thread(target=self._voice_shutdown, daemon=True).start()
                return "shutting down"
            self._followup_until = time.monotonic() + self._followup_window
            self.prefetch.after_turn(outcome.category, outcome.escalated, outcome.text)
            return outcome.spoke
        except Exception as exc:
            self.bus.log(f"typed turn failed: {exc}", "error")
            return f"that failed: {exc}"
        finally:
            self._turn_lock.release()
            # Hand the floor back. Ordering matters: the queue must not be told
            # it may resume until the lock is genuinely free, or it wakes up and
            # immediately blocks again on a lock this thread still holds.
            if self.query is not None:
                self.query.release_pause()
            self.bus.set_state(self._resting_state())

    def _arm_watchdog(self, sw: Stopwatch) -> threading.Timer:
        """Never let a turn go silent.

        The latency filler covers the first second. This covers everything after
        it: a model that stalls, a skill that blocks on I/O, a player that does
        not answer. Without it a slow turn is indistinguishable from a crash —
        which is exactly how "play my playlist" felt when it froze.
        """
        def bark() -> None:
            if self.speaker.speaking or self._closing.is_set():
                return
            self.bus.log(f"turn still running at {sw.elapsed_ms:.0f}ms", "warn")
            self.speaker.say_line("hold")

        t = threading.Timer(float(self.cfg.get("latency.watchdog_s", 3.5)), bark)
        t.daemon = True
        t.start()
        return t

    def _run_turn(self, utt: Utterance) -> None:
        sw = Stopwatch(self.bus, "turn")
        watchdog = None
        try:
            self.bus.set_state(State.TRANSCRIBING)
            text = self.recognizer.transcribe(utt.audio)
            sw.mark("asr")
            if not text:
                self.bus.set_state(self._resting_state())
                return

            addressed, remainder = detect_wake(text, self.cfg)
            in_window = time.monotonic() < self._followup_until

            if not addressed and not in_window:
                # Heard, understood, not for us. Log it so the HUD shows the
                # system is alive and listening without acting on stray speech.
                self.bus.log(f"(not addressed) {text[:60]}", "heard")
                self.bus.set_state(State.DORMANT)
                return

            payload = remainder if addressed else text
            self.bus.publish(TRANSCRIPT_FINAL, text=text, addressed=True)
            # The name on its own has no payload to echo, and logging an empty
            # "> " reads like a dropped transcript rather than a deliberate
            # silent wake.
            self.bus.log(f"> {payload[:70]}" if payload.strip()
                         else "(awake, listening)", "user")

            watchdog = self._arm_watchdog(sw)
            outcome = self.callosum.handle(payload, sw)

            if outcome.shutdown:
                self.bus.log("shutdown requested by voice", "boot")
                threading.Thread(target=self._voice_shutdown, daemon=True).start()
                return
            if outcome.sleep:
                self._followup_until = 0.0
                self.bus.set_state(State.DORMANT)
                return

            if self._followup_extends:
                self._followup_until = time.monotonic() + self._followup_window

            # The user is now listening to the answer. Spend that window
            # predicting and pre-loading for the next turn.
            self.prefetch.after_turn(outcome.category, outcome.escalated, outcome.text)

            first_audio = next((ms for n, ms in sw.marks if n == "first_audio"), None)
            if first_audio is None:
                first_audio = next((ms for n, ms in sw.marks if n == "first_token"), None)
            if self._warn_budget and first_audio and first_audio > self._budget_ms:
                self.bus.log(f"over budget: {sw.breakdown()}", "warn")
            elif first_audio:
                self.bus.log(f"turn {first_audio:.0f}ms to audio", "metric")

            self.bus.set_state(self._resting_state())
        except Exception as exc:
            # A failed turn must still produce a sound. Dying silently is the one
            # failure mode the operator cannot distinguish from being ignored.
            self.bus.log(f"turn failed: {type(exc).__name__}: {exc}", "error")
            self.bus.set_state(State.ALERT)
            try:
                self.speaker.say_line("veto.error")
            except Exception:
                pass
            self.bus.set_state(State.DORMANT)
        finally:
            if watchdog is not None:
                watchdog.cancel()
            self._turn_lock.release()
            if self.query is not None:
                self.query.release_pause()

    def _resting_state(self) -> State:
        return State.LISTENING if time.monotonic() < self._followup_until else State.DORMANT

    def _voice_shutdown(self) -> None:
        # The farewell is a courtesy; quitting is the instruction. Never let a
        # failure in the former prevent the latter — that is how "HIVE, poweroff"
        # ends up doing nothing at all.
        try:
            self.speaker.say_line("farewell")
        except Exception as exc:
            self.bus.log(f"farewell failed: {exc}", "warn")
        self.request_quit()

    # Replaced by the UI so a voice shutdown closes the window too.
    def request_quit(self) -> None:
        self.stop()

    # -- consolidation ------------------------------------------------------
    def save_transcript(self) -> str:
        """Put the conversation in the vault right now. No model, no waiting.

        Milliseconds, because everything it writes already exists: the turns,
        the actions, the salient terms. Called first thing in consolidate() and
        again by the `save` command, and it is the entire guarantee — from the
        moment this returns, the session is on disk whatever happens next.

        Returns the note's name, or '' if there was nothing to write. Never
        raises: a caller on the way out has nothing useful to do with an
        exception, and the one after this must still run.
        """
        if self.stm.user_turns == 0:
            return ""
        try:
            note = self.ltm.write_session(
                self.stm, self.callosum.actions_log,
                "_Summary pending — the transcript below is complete._",
                self._fallback_topics(),
                calibrations=self.calibrator.stats())
            self.bus.publish(SESSION_SAVED, stage="transcript",
                             name=note.path.name, path=str(note.path),
                             verified=note.path.exists())
            return note.path.name
        except Exception as exc:
            self.bus.log(f"transcript save failed: {type(exc).__name__}: {exc}",
                         "error")
            return ""

    def save_now(self) -> str:
        """`save` in the terminal. Write the transcript and verify it on disk.

        The point is the verification. Anyone can write a file and say they
        wrote a file; this reads it back and reports what is actually there, so
        the answer to "is my conversation in Obsidian" stops being a matter of
        trust and starts being a receipt.
        """
        name = self.save_transcript()
        if not name or self.ltm.session_note is None:
            return ""
        receipt = self._verify_note(self.ltm.session_note)
        if not receipt:
            return "the note was written but could not be verified — see .logs/hive.log"
        size = self.ltm.session_note.stat().st_size
        return (f"saved and verified on disk:\n"
                f"  {self.ltm.session_note}\n"
                f"  {receipt.split(' · ')[1]} · {size:,} bytes · "
                f"{self.stm.user_turns} exchange(s)\n"
                f"  the summary is written over this on shutdown; the "
                f"transcript is already complete.")

    def consolidate(self) -> None:
        """Write the session to Obsidian. Runs once, on the way out.

        Two passes over one note, and the order is the fix for the defect this
        method existed with for six versions.

            1. the transcript, immediately, with no model involved
            2. the summary and the topic notes, over the top of it

        Pass two calls the heavy model with an 1800-token budget, which the log
        measures at nine to fifty-seven seconds. Everything about the shutdown
        path used to hang off that call finishing: the note was written *after*
        it, so a process that died during it — and seven sessions in
        .logs/hive.log did — lost a conversation that had been complete in
        memory the whole time. Now the worst that a kill during pass two can
        cost is the summary paragraph, and the transcript is already there.

        The write is verified by reading the file back rather than by assuming
        the call worked, which is the same rule `_plan_document` follows: HIVE
        does not report a file it has not seen on disk.
        """
        if not bool(self.cfg.get("session.consolidate_on_exit", True)):
            return
        if self.stm.user_turns == 0:
            self.bus.log("nothing to consolidate", "memory")
            self.subjective.save()
            return

        self.bus.log("consolidating session to vault", "memory")

        # -- pass one: the transcript, now ----------------------------------
        # Normally already done — stop() does it before any teardown step that
        # can block. Repeated here so consolidate() is correct on its own, which
        # is how --dryrun calls it, and skipped when the note already exists so
        # the log does not carry two identical lines. That log is the diagnostic
        # for this whole class of failure; it should not be ambiguous.
        if self.ltm.session_note is None:
            name = self.save_transcript()
            if name:
                self.bus.log(f"transcript safe in {name}", "memory")

        # -- pass two: the summary, over the top ----------------------------
        summary, topics = self._summarize()
        try:
            note = self.ltm.write_session(
                self.stm, self.callosum.actions_log, summary, topics,
                calibrations=self.calibrator.stats())
            link = self.ltm.link_for(note)
            # Three at most. Five produces a topic note for every passing remark,
            # and a folder of single-entry notes is noise, not memory.
            for topic in topics[:3]:
                entry = self._topic_entry(topic, summary)
                self.ltm.append_topic(topic, entry, link)
            self.last_note = self._verify_note(note.path)
            self.bus.log(f"wrote {note.path.name}", "memory")
        except Exception as exc:
            self.bus.log(f"consolidation failed: {type(exc).__name__}: {exc}",
                         "error")
            # Pass one may still have landed. Report on the file rather than on
            # the exception — "the summary failed" and "the session was lost"
            # are very different pieces of news.
            if self.ltm.session_note is not None:
                self.last_note = self._verify_note(self.ltm.session_note)

        self.subjective.observe_salience(self.stm)
        self.subjective.model.voice = self.speaker.voice
        self.subjective.save()

    def _verify_note(self, path) -> str:
        """Read the note back off disk. Returns a receipt, or '' if it is not there.

        The difference between believing the write worked and knowing it did.
        Everything upstream can lie by omission — an exception swallowed three
        frames down, a filesystem that reported success and did nothing — and
        the whole complaint this addresses is that checking the folder by hand
        was the only way to find out.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            self.bus.log(f"the note could not be read back: {exc}", "error")
            return ""
        if "## Transcript" not in text:
            self.bus.log(f"{path.name} is on disk but has no transcript", "error")
            return ""
        self.vault_receipt = (f"{path.name} · {len(text.splitlines())} lines · "
                              f"{path.parent}")
        self.bus.log(f"verified {self.vault_receipt}", "memory")
        self.bus.publish(SESSION_SAVED, stage="summary", name=path.name,
                         path=str(path), verified=True)
        return self.vault_receipt
        # Fold this session's note into the association index while we are still
        # here. Doing it now means the connection is available on the next
        # session's first turn rather than its second.
        try:
            self.associations.build()
        except Exception as exc:
            self.bus.log(f"association rebuild failed: {exc}", "warn")

    def _summarize(self) -> tuple[str, list[str]]:
        transcript = "\n".join(
            f"{'Jay' if t.role == 'user' else 'HIVE'}: {t.text}"
            for t in self.stm.transcript() if t.text
        )[-6000:]
        if not self.llm.available():
            return self._fallback_summary(), self._fallback_topics()

        use_heavy = self.cfg.get("memory.consolidation_model", "heavy") == "heavy"
        prompt = [
            {"role": "system", "content":
                "You summarise a voice session for an Obsidian vault. Write markdown "
                "prose, not speech. Be specific and concrete — name the actual things "
                "discussed. No preamble, no meta-commentary about the transcript."},
            {"role": "user", "content":
                f"Session transcript:\n\n{transcript}\n\n"
                "Write exactly two parts.\n"
                # Was "three to six sentences", and every topic note in the vault
                # is thirteen lines long as a direct result: the topic entry is
                # one sentence lifted from here, so a thin summary produces a
                # thin note that will still be thin in six months.
                "PART 1: several paragraphs covering what was discussed and what "
                "was decided. Go subject by subject. Name the specifics — the "
                "actual terms, numbers, conclusions and reasons. Include things "
                "that came up once and were not pursued. Length is not a "
                "problem here; leaving something out is.\n"
                "PART 2: a line starting with 'TOPICS:' followed by two or three "
                "topic names, comma separated.\n"
                "Topic names become note filenames in an Obsidian vault, so:\n"
                "  - Title Case With Real Spaces Between Words. Never run words "
                "together like 'DiskSpaceCheck' or 'diskspacecheck'.\n"
                "  - Two to four words each. No punctuation.\n"
                "  - Name durable subjects a person would file under (\"eJPT "
                "Preparation\"), not one-off events (\"Asked About Disk Space\")."},
        ]
        try:
            raw = self.llm.complete(prompt, heavy=use_heavy, num_predict=1800)
        except Exception:
            return self._fallback_summary(), self._fallback_topics()

        summary, topics = raw, []
        if "TOPICS:" in raw:
            head, _, tail = raw.partition("TOPICS:")
            summary = head.strip()
            topics = [t.strip(" .#*-") for t in tail.split(",") if t.strip()]
        # The model echoes the scaffolding it was given. Strip it rather than
        # prompt harder — small models will keep doing this.
        summary = re.sub(r"(?im)^\s*PART\s*\d+\s*[:.\-]?\s*", "", summary)
        summary = re.sub(r"(?i)\bPART\s*\d+\s*:\s*$", "", summary).strip()

        # Three topics, decided here so the session's wikilinks and the topic
        # notes that actually get written can never disagree.
        topics = [slugify_title(t) for t in topics if 2 < len(t) < 48][:3]
        return (clean_for_speech(summary) or self._fallback_summary(),
                topics or self._fallback_topics())

    def _fallback_summary(self) -> str:
        mins = self.stm.duration / 60.0
        acts = ", ".join(self.callosum.actions_log[:6]) or "no actions"
        return (f"Voice session of {mins:.0f} minutes over {self.stm.user_turns} "
                f"exchanges. Actions: {acts}.")

    def _fallback_topics(self) -> list[str]:
        return [slugify_title(t) for t, _ in self.stm.salient(3)] or ["General"]

    def _topic_entry(self, topic: str, summary: str) -> str:
        """The paragraph a topic note accumulates from this session.

        Was one sentence — the first in the summary that happened to contain the
        topic's first word — which is why every topic note in this vault is
        thirteen lines long and says almost nothing. A topic note is supposed to
        be the thing you read in three months to remember what was established,
        and one sentence per session cannot carry that.

        So: every sentence that bears on the topic, not the first one, and the
        opening of the summary as a floor when none of them do. Still bounded,
        because a topic note that is a second copy of the session note is not a
        topic note either — the session already holds the transcript.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        if not sentences:
            return ""
        words = [w for w in re.split(r"\W+", topic.lower()) if len(w) > 3]
        hits = [s for s in sentences
                if any(w in s.lower() for w in words)] if words else []
        if not hits:
            hits = sentences[:3]
        entry = " ".join(hits[:6])
        return entry if len(entry) <= 1400 else entry[:1400].rsplit(" ", 1)[0] + "…"
