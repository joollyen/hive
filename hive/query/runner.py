"""The query runner — HIVE working down a written list on its own.

HIVE's turn loop is built on one invariant, stated at the top of runtime.py:
exactly one turn runs at a time, and an utterance arriving mid-turn is dropped
rather than queued. That invariant is what makes the whole system predictable,
and this module does not break it. A query task takes the same lock a spoken turn
takes and runs through the same callosum, so from everything downstream a task is
indistinguishable from Jay having typed it. That is the point: the answer he
hears is the answer he would have got by asking.

What is genuinely new is everything around that:

**The watchdog is on inactivity, not duration.** A ten-minute research task that
is streaming sentences is working; a thirty-second one that has gone silent is
stuck. So the clock resets on every sign of life — a synthesis unit reaching the
speakers, a stopwatch mark, a state change, a plan step, results coming back off
the network — and it is the *gap* between those that expires. A fixed deadline
would kill the long tasks that are the reason for having a queue at all.

**A stalled task cannot be killed, only abandoned.** Python has no way to stop a
running thread, so the honest design is to stop *waiting* for it: interrupt the
speaker, give it a grace period to unwind on its own, mark it incomplete, and
move on. If it is still alive after the grace period the query ends rather than
starting the next task on top of it — running two turns at once is exactly the
invariant above, and breaking it quietly to keep a queue moving would trade a
visible failure for an invisible one.

**Every task writes to the vault, including the ones that failed.** That is the
difference between a queue and a script. A failed task's note records the reason
and whatever it did manage before it stopped, because the thing Jay actually
wants back from an unattended run is not the successes — those spoke for
themselves — it is a list of what to go and fix.

**A live request wins.** Speaking to HIVE while the query runs pauses it at the
next task boundary. The queue is background work by construction and the person
in the room is not.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime

from ..events import (METRIC, PLAN_PROGRESS, QUERY_OUTPUT, QUERY_STATE, QUERY_TASK,
                      SPEAK_CHUNK, SPEAK_START, STATE_CHANGED, WEB_RESULTS,
                      EventBus, Stopwatch)
from .tasklist import (DONE, INCOMPLETE, PENDING, RUNNING, SKIPPED, QueryTask,
                       parse, summarise)

# Every event that counts as "still working". Deliberately a list of specific
# events rather than a wildcard subscription: a wildcard would be stamped by
# HIVE's own periodic housekeeping — the night-mode timer, an association
# rebuild — and a watchdog that a background timer can satisfy is not a watchdog.
_HEARTBEATS = (SPEAK_CHUNK, SPEAK_START, STATE_CHANGED, METRIC, PLAN_PROGRESS,
               WEB_RESULTS)

_ORDINALS = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
             "Nine", "Ten"]

# A leading wake word, removed the way the microphone path removes it — but by
# hand rather than through `detect_wake`, and that difference is load-bearing.
# detect_wake returns the *normalised* remainder, and normalisation strips every
# character outside \w\s'&/.:- — which deletes the plus in "betterminecraft+
# hardcore gameplay". A query is typed, so its punctuation was chosen rather
# than transcribed, and it has to cross intact.
_LEAD_WAKE = re.compile(r"^\s*(?:hey\s+|ok\s+|okay\s+)?h(?:ive)?\b[\s,:.\-]*", re.I)


class QueryRunner:
    def __init__(self, cfg, bus: EventBus, runtime):
        self.cfg, self.bus, self.runtime = cfg, bus, runtime

        self.watchdog_s = float(cfg.get("query.stall_after_s", 30.0))
        # An absolute ceiling, off by default. Inactivity is the right measure
        # and this is deliberately not a substitute for it: a research task
        # measured at 107 seconds on this box was working the whole time, and a
        # flat thirty-second cap would have filed it as incomplete after reading
        # out three of its four papers. This exists for the case where that
        # tradeoff is not wanted — set it and no task can outrun it, whatever it
        # is doing.
        self.max_task_s = float(cfg.get("query.max_task_s", 0.0))
        self.grace_s = float(cfg.get("query.unwind_grace_s", 20.0))
        self.gap_s = float(cfg.get("query.gap_between_tasks_s", 1.2))
        self.lock_wait_s = float(cfg.get("query.lock_wait_s", 120.0))
        self.pause_wait_s = float(cfg.get("query.pause_wait_s", 180.0))
        self._announce_start = bool(cfg.get("query.announce_start", True))
        self._announce_end = bool(cfg.get("query.announce_end", True))
        self._narrate_incomplete = bool(cfg.get("query.narrate_incomplete", True))

        self.tasks: list[QueryTask] = []
        self.buffer: str = ""
        self.session_id = ""
        self.stop_reason = ""

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._pause_reason = ""
        self._state_lock = threading.RLock()
        self._activity = 0.0
        self._current: QueryTask | None = None
        self._speech: list[str] = []
        self._results: list[dict] = []
        self._orphaned = False

        for name in _HEARTBEATS:
            bus.subscribe(name, self._touch)
        bus.subscribe(SPEAK_CHUNK, self._collect_speech)
        bus.subscribe(WEB_RESULTS, self._collect_results)

    # -- state ---------------------------------------------------------------
    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def _touch(self, _evt=None) -> None:
        self._activity = time.monotonic()

    def _collect_speech(self, evt) -> None:
        """Mirror HIVE's own speech into the running task's record.

        This is the only complete account of what a task produced. `Outcome.spoke`
        exists too, but a task that stalls never returns one — and a stalled task
        is precisely the case where "what did it manage before it stopped" is the
        whole value of the note.
        """
        if self._current is None:
            return
        text = (evt.payload.get("text") or "").strip()
        if text:
            self._speech.append(text)
            self.bus.publish(QUERY_OUTPUT, text=text, kind="speech")

    def _collect_results(self, evt) -> None:
        if self._current is None:
            return
        for r in evt.payload.get("results") or []:
            self._results.append(dict(r))

    def snapshot(self) -> dict:
        """Everything the HIVE-Q window draws, in one read."""
        with self._state_lock:
            cur = self._current
            return {
                "running": self.running,
                "paused": self.paused,
                "pause_reason": self._pause_reason,
                "stopping": self._stop.is_set(),
                "index": cur.index if cur else 0,
                "total": len(self.tasks),
                "current": cur.short() if cur else "",
                "elapsed": cur.duration if cur else 0.0,
                "stall_in": max(0.0, self.watchdog_s - (
                    time.monotonic() - self._activity)) if cur else 0.0,
                "tasks": [self._as_dict(t) for t in self.tasks],
            }

    @staticmethod
    def _as_dict(t: QueryTask) -> dict:
        return {"index": t.index, "text": t.text, "title": t.title,
                "status": t.status, "reason": t.reason, "spoken": t.spoken,
                "note": t.note, "duration": round(t.duration, 1),
                "results": len(t.results), "line": t.line}

    def _publish_state(self) -> None:
        snap = self.snapshot()
        self.bus.publish(QUERY_STATE, **{k: v for k, v in snap.items()
                                         if k != "tasks"})

    def _publish_task(self, task: QueryTask) -> None:
        self.bus.publish(QUERY_TASK, total=len(self.tasks), **self._as_dict(task))

    def say_out(self, text: str, kind: str = "info") -> None:
        self.bus.publish(QUERY_OUTPUT, text=text, kind=kind)

    # -- lifecycle -----------------------------------------------------------
    def start(self, buffer: str | None = None) -> tuple[bool, str]:
        """Parse the buffer and begin. Returns (started, message)."""
        if self.running:
            return False, ("A query's already running — say or type `end query` "
                           "to stop it first.")
        if buffer is not None:
            self.buffer = buffer
        tasks = parse(self.buffer)
        if not tasks:
            return False, "There's nothing in the query to run."

        with self._state_lock:
            self.tasks = tasks
            self._current = None
            self._orphaned = False
            self.stop_reason = ""
        self._stop.clear()
        self._resume.set()
        self._pause_reason = ""
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._touch()

        self._thread = threading.Thread(target=self._run, name="hive-query",
                                        daemon=True)
        self._thread.start()
        return True, f"Query started — {len(tasks)} task(s)."

    def end(self, why: str = "ended by hand") -> str:
        """The override. Stops the current task and stands the query down.

        Reachable from the HIVE terminal (`end query`), by voice, and from the
        HIVE-Q window, and it does the same thing from all three: cut the speech
        immediately, stop waiting on whatever is running, and leave the untouched
        tasks in the queue so the list can be resumed rather than retyped.
        """
        if not self.running:
            return "No query is running."
        self.stop_reason = why
        self._stop.set()
        self._resume.set()            # a paused runner has to wake up to exit
        try:
            self.runtime.speaker.interrupt()
        except Exception:
            pass
        self.bus.log(f"query ended: {why}", "action")
        self._publish_state()
        return "Ending the query. Stopping what's running and standing down."

    def request_pause(self, why: str = "a live request") -> None:
        """Hold the queue at the next task boundary. Called by the runtime when
        Jay speaks or types while the query is working."""
        if not self.running:
            return
        if self._resume.is_set():
            self.bus.log(f"query paused for {why}", "action")
            self.say_out(f"— holding for {why} —", "pause")
        self._pause_reason = why
        self._resume.clear()
        self._publish_state()

    def release_pause(self) -> None:
        if not self._resume.is_set():
            self._pause_reason = ""
            self._resume.set()
            self._publish_state()

    def stop(self) -> None:
        """Session shutdown. Never blocks the vault write on the way out."""
        if self.running:
            self.end("the session closed")
            t = self._thread
            if t is not None:
                t.join(min(10.0, self.grace_s))

    # -- the loop ------------------------------------------------------------
    def _run(self) -> None:
        total = len(self.tasks)
        self.bus.log(f"query running: {total} task(s)", "action")
        self.say_out(f"Query started — {total} task(s).", "sys")
        self._publish_state()

        if self._announce_start:
            word = _ORDINALS[total] if total < len(_ORDINALS) else str(total)
            self._speak(f"{word} in the query. Working through them now.")

        for task in self.tasks:
            if self._stop.is_set():
                break
            if not self._await_resume():
                break
            if self._stop.is_set():
                break

            self._run_task(task)
            self._write_note(task)
            self._publish_task(task)
            self._publish_state()

            if self._orphaned:
                # A task that never unwound. Starting the next one would put two
                # turns through the callosum at once, which is the one thing the
                # whole runtime is arranged to prevent.
                self._stop.set()
                self.stop_reason = "a task wouldn't stop"
                break
            if self._stop.is_set():
                break
            time.sleep(self.gap_s)

        for task in self.tasks:
            if task.status in (PENDING, RUNNING):
                task.status = SKIPPED
                task.reason = (f"the query ended before it ran — "
                               f"{self.stop_reason}" if self.stop_reason
                               else "the query ended before it ran")
                self._publish_task(task)

        self._finish()

    def _await_resume(self) -> bool:
        """Block while a live request has the floor. False means give up."""
        if self._resume.is_set():
            return True
        self._publish_state()
        got = self._resume.wait(self.pause_wait_s)
        self._touch()
        if not got:
            self.bus.log("query gave up waiting to resume", "warn")
        return got and not self._stop.is_set()

    def _finish(self) -> None:
        with self._state_lock:
            self._current = None
        line = summarise(self.tasks)
        if self._orphaned:
            line += (" I stopped there rather than start the next one on top of "
                     "a task that hadn't finished.")
        self.bus.log(f"query finished: {line}", "action")
        self.say_out(line, "sys")
        if self._announce_end:
            self._speak(line)
        self._publish_state()

    # -- one task ------------------------------------------------------------
    def _run_task(self, task: QueryTask) -> None:
        with self._state_lock:
            self._current = task
            self._speech = []
            self._results = []
        task.status = RUNNING
        task.started = time.time()
        self._touch()
        self.bus.log(f"query {task.index}/{len(self.tasks)}: {task.short()}", "action")
        self.say_out(f"▸ {task.index}. {task.text}", "task")
        self._publish_task(task)
        self._publish_state()

        actions = self.runtime.callosum.actions_log
        before = len(actions)

        # The same lock a spoken turn takes. Waiting rather than dropping,
        # because unlike an utterance a queued task has nowhere else to go.
        if not self.runtime._turn_lock.acquire(timeout=self.lock_wait_s):
            self._settle(task, INCOMPLETE,
                         "HIVE was busy with something else the whole time")
            return

        # Silence the curiosity layer for the duration. It volunteers one
        # connection or follow-up question per turn, which is right in a
        # conversation and wrong here — a query runs unattended by definition,
        # so "anything in research that hasn't clicked yet?" is a question asked
        # of an empty room, and it lands in the task's note as though it were
        # part of the answer. Observed on the first real run.
        curiosity = getattr(self.runtime, "curiosity", None)
        was_curious = getattr(curiosity, "enabled", False)
        if curiosity is not None:
            curiosity.enabled = False

        holder: dict = {}
        worker = threading.Thread(target=self._execute, args=(task, holder),
                                  name="hive-query-task", daemon=True)
        self._touch()
        worker.start()
        verdict = self._watch(worker)

        if verdict != "finished":
            # Cut the speech first: say() and say_stream() both check the
            # interrupt flag, so this is what actually unblocks a worker that is
            # mid-answer rather than mid-thought.
            try:
                self.runtime.speaker.interrupt()
            except Exception:
                pass
            worker.join(self.grace_s)

        alive = worker.is_alive()
        if alive:
            self._orphaned = True

        if curiosity is not None:
            curiosity.enabled = was_curious
        try:
            self.runtime._turn_lock.release()
        except RuntimeError:
            pass

        task.actions = list(actions[before:])
        task.results = list(self._results)
        task.spoken = " ".join(self._speech).strip()

        if verdict == "finished":
            error = holder.get("error")
            refused = holder.get("refused")
            if error:
                self._settle(task, INCOMPLETE, f"it failed: {error}")
            elif refused:
                self._settle(task, INCOMPLETE, refused)
            else:
                self._settle(task, DONE, "")
        elif verdict == "stopped":
            self._settle(task, INCOMPLETE,
                         "you ended the query while it was running"
                         if self.stop_reason in ("", "ended by hand")
                         else f"the query was stopped — {self.stop_reason}")
        elif verdict == "overran":
            self._settle(task, INCOMPLETE,
                         f"it ran past the {int(self.max_task_s)} second ceiling "
                         f"on a single task")
        else:   # stalled
            secs = int(self.watchdog_s)
            self._settle(task, INCOMPLETE,
                         f"no progress for {secs} seconds, so I stopped waiting "
                         f"on it" + (" — and it still hadn't stopped when I "
                                     "moved on" if alive else ""))

    def _execute(self, task: QueryTask, holder: dict) -> None:
        """Run the task exactly as a typed request, on a thread we can abandon."""
        sw = Stopwatch(self.bus, "query")
        try:
            payload = _LEAD_WAKE.sub("", task.text).strip() or task.text
            outcome = self.runtime.callosum.handle(payload, sw)
            # A queue must not be able to power the machine off. Shutting down
            # is a decision made in the room, not one that can be buried on line
            # four of a list typed twenty minutes earlier.
            if outcome.shutdown:
                holder["refused"] = ("it asked me to shut down, which I won't do "
                                     "from inside a query")
            elif outcome.sleep:
                holder["refused"] = "it told me to stand down, which a query can't"
            elif outcome.failed:
                # The action ran, said what went wrong out loud, and did not
                # work. In a conversation that is the end of it; in a queue it
                # is the whole reason the incomplete tab exists.
                holder["refused"] = outcome.failure or "the action didn't work"
            holder["outcome"] = outcome
            self.runtime.prefetch.after_turn(outcome.category, outcome.escalated,
                                             outcome.text)
        except Exception as exc:
            holder["error"] = f"{type(exc).__name__}: {exc}"
            self.bus.log(f"query task {task.index} raised: {exc}", "error")
        finally:
            holder["done"] = True
            self._touch()

    def _watch(self, worker: threading.Thread) -> str:
        """Wait for the task, watching for silence. -> finished|stalled|stopped.

        HIVE talking counts as progress, and this is not a refinement — it is
        the difference between the watchdog working and the watchdog being
        actively wrong. `say()` publishes a chunk as each synthesis unit is
        *queued*, then blocks in the playback drain until the audio has actually
        been heard. A four-paragraph research answer therefore emits all its
        events in the first few seconds and then plays for forty more with the
        bus completely silent. Measured on a real run: a task that had already
        found and read out four papers was declared stalled at the thirty-second
        mark and filed as incomplete. A task cannot be stuck while HIVE is
        mid-sentence, so the speaker's own state is checked directly rather than
        inferred from events it does not emit.
        """
        began = time.monotonic()
        while worker.is_alive():
            if self._stop.is_set():
                return "stopped"
            if self.max_task_s and time.monotonic() - began > self.max_task_s:
                return "overran"
            if getattr(self.runtime.speaker, "speaking", False):
                self._touch()
            elif time.monotonic() - self._activity > self.watchdog_s:
                return "stalled"
            worker.join(0.2)
        return "finished"

    def _settle(self, task: QueryTask, status: str, reason: str) -> None:
        task.status = status
        task.reason = reason
        task.finished = time.time()
        with self._state_lock:
            self._current = None
        if status == DONE:
            self.say_out(f"✓ {task.index}. done in {task.duration:.0f}s", "done")
            return
        self.say_out(f"✗ {task.index}. incomplete — {reason}", "incomplete")
        self.bus.log(f"query {task.index} incomplete: {reason}", "warn")
        if self._narrate_incomplete and not self._stop.is_set():
            # Said out loud, because an unattended queue that fails quietly is
            # a queue you have to go and read to discover it failed.
            self._speak(f"That one didn't go through — {reason}. "
                        f"It's in the incomplete tab.")

    def _speak(self, line: str) -> None:
        try:
            self.runtime.speaker.say(line, cached=False)
        except Exception as exc:
            self.bus.log(f"query speech failed: {exc}", "warn")
        self._touch()

    # -- the vault -----------------------------------------------------------
    def _write_note(self, task: QueryTask) -> None:
        """One markdown file per task, after every task, pass or fail."""
        try:
            note = self.runtime.ltm.write_query_task(
                task, session_id=self.session_id, total=len(self.tasks))
            task.note = note.path.name
            self.bus.log(f"query note: {note.path.name}", "memory")
            self.say_out(f"   logged to query_tasks/{note.path.name}", "note")
        except Exception as exc:
            # The note failing must not take the queue down with it — the work
            # was done either way, and the next task is still worth running.
            self.bus.log(f"query note failed for task {task.index}: {exc}", "error")
            self.say_out(f"   couldn't write the note: {exc}", "incomplete")
