"""Bridge between the engine threads and the QML scene.

Qt properties may only be touched on the thread that owns the object, but events
arrive from the PortAudio callbacks, the VAD thread and the turn thread. So every
inbound event is re-emitted as a private Signal; Qt's automatic connection type
queues those across threads and delivers them on the GUI thread. Nothing here
writes a property directly from a foreign thread.

The audio spectrum is decimated from 128 analysis bands to the 64 the shader
consumes, and exposed as a flat list that QML packs into 16 vec4 uniforms.

The terminal's backlog is a real QAbstractListModel rather than a list property,
for reasons documented on TerminalModel — it is the difference between typing
into a responsive pane and typing into one that hitches on every line of output.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from PySide6.QtCore import (Property, QAbstractListModel, QByteArray, QModelIndex,
                            QObject, Qt, Signal, Slot)

from ..events import (AUDIO_LEVEL, CALIBRATION, ENGINE_READY, LOG_LINE,
                      SESSION_SAVED, SPEAK_CHUNK, SPEAK_END, SPEAK_START,
                      STATE_CHANGED, STATUS_WORD, TERMINAL_OUTPUT,
                      TERMINAL_TOGGLE, TRANSCRIPT_FINAL, EventBus, State)

SHADER_BANDS = 64
MAX_LOG = 5
MAX_TERM = 400


class TerminalModel(QAbstractListModel):
    """Incremental backlog for the terminal pane.

    The first version exposed the whole backlog as a plain `list` property. Every
    appended line therefore replaced the model wholesale, and QML responded by
    destroying and rebuilding every delegate in view — re-running each one's
    entrance animation as it went. Output arrived in a visible shudder, and
    keystrokes landed behind it because the same thread was rebuilding delegates.

    Appending a row here touches exactly one delegate. That is the whole fix.
    """

    LineRole = Qt.UserRole + 1
    KindRole = Qt.UserRole + 2

    def __init__(self, maxlen: int = MAX_TERM, parent=None):
        super().__init__(parent)
        self._rows: list[tuple[str, str]] = []
        self._max = maxlen

    # -- QAbstractListModel -------------------------------------------------
    def roleNames(self) -> dict:
        return {self.LineRole: QByteArray(b"line"),
                self.KindRole: QByteArray(b"kind")}

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role: int = LineRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        line, kind = self._rows[index.row()]
        if role == self.LineRole:
            return line
        if role == self.KindRole:
            return kind
        return None

    # -- mutation (GUI thread only) -----------------------------------------
    def append(self, text: str, kind: str = "out") -> None:
        lines = (text or "").split("\n") or [""]
        first = len(self._rows)
        self.beginInsertRows(QModelIndex(), first, first + len(lines) - 1)
        self._rows.extend((line, kind) for line in lines)
        self.endInsertRows()
        self._trim()

    def extend_last(self, more: str) -> None:
        """Grow the final row in place — how streamed speech lands in the pane."""
        if not self._rows:
            self.append(more)
            return
        row = len(self._rows) - 1
        line, kind = self._rows[row]
        self._rows[row] = (line + more, kind)
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx, [self.LineRole])

    def clear(self) -> None:
        if not self._rows:
            return
        self.beginResetModel()
        self._rows.clear()
        self.endResetModel()

    # -- reading back out ----------------------------------------------------
    # Selection lives in QML as a pair of row indices; turning that pair into
    # text has to happen here, because QML can see one delegate at a time and
    # the rows it wants are mostly not instantiated.
    def text_range(self, first: int, last: int) -> str:
        lo = max(0, min(first, last))
        hi = min(len(self._rows) - 1, max(first, last))
        if hi < lo:
            return ""
        return "\n".join(line for line, _ in self._rows[lo:hi + 1])

    def block_at(self, row: int) -> tuple[int, int]:
        """The whole exchange around a row: one typed line and its answer.

        What a double-click should select. A wrapped answer is a dozen rows in
        this model and picking them off one at a time with a drag — in a pane
        that is still scrolling as HIVE talks — is the kind of thing you only
        do once before going back to not copying anything.
        """
        if not 0 <= row < len(self._rows):
            return (row, row)
        lo = row
        while lo > 0 and self._rows[lo][1] != "in":
            lo -= 1
        hi = row
        while hi + 1 < len(self._rows) and self._rows[hi + 1][1] != "in":
            hi += 1
        return (lo, hi)

    def _trim(self) -> None:
        excess = len(self._rows) - self._max
        if excess <= 0:
            return
        self.beginRemoveRows(QModelIndex(), 0, excess - 1)
        del self._rows[:excess]
        self.endRemoveRows()


class UIBridge(QObject):
    # -- notifications consumed by QML --------------------------------------
    spectrumChanged = Signal()
    levelChanged = Signal()
    stateChanged = Signal()
    statusWordChanged = Signal()
    transcriptChanged = Signal()
    logChanged = Signal()
    readyChanged = Signal()
    speakingChanged = Signal()
    terminalOpenChanged = Signal()
    terminalLinesChanged = Signal()
    terminalBusyChanged = Signal()
    historyChanged = Signal()
    clipChanged = Signal()
    closingChanged = Signal()
    vaultChanged = Signal()

    # -- private cross-thread carriers --------------------------------------
    _sigAudio = Signal(float, list, str)
    _sigState = Signal(str)
    _sigStatus = Signal(str)
    _sigTranscript = Signal(str)
    _sigLog = Signal(str, str)
    _sigReady = Signal()
    _sigSpeaking = Signal(bool)
    _sigTermOpen = Signal(bool)
    _sigTermOut = Signal(str, str)
    _sigTermBusy = Signal(bool)
    _sigSpokenChunk = Signal(str, bool)
    _sigClip = Signal()
    _sigVault = Signal(str, str, bool)
    _sigClosing = Signal(bool)

    def __init__(self, bus: EventBus, parent=None):
        super().__init__(parent)
        self.bus = bus
        self._spectrum = [0.0] * SHADER_BANDS
        self._level = 0.0
        self._state = State.DORMANT.value
        self._statusWord = ""
        self._transcript = ""
        self._log: deque[str] = deque(maxlen=MAX_LOG)
        self._ready = False
        self._speaking = False
        self._term_open = False
        self._term_busy = False
        self._term_model = TerminalModel(MAX_TERM, self)
        self._term_exec = None
        self._vocab_fn = None
        self._draft_cb = None
        self._history: list[str] = []
        self._t0 = time.monotonic()
        self._quit_cb = None
        self._clipboard = None
        self._clippings = None
        self._copy_fn = None
        # The shutdown veil, and what consolidation has managed so far. Closing
        # HIVE writes the session to Obsidian and that can take the better part
        # of a minute; a window that looks frozen for a minute gets force-quit,
        # and a force-quit during consolidation is how sessions were lost.
        self._closing = False
        self._vault_stage = ""
        self._vault_name = ""
        # Set while a spoken reply is being mirrored into the pane, so successive
        # clauses extend one line instead of stacking one line per clause.
        self._spoken_open = False
        self._spoken_count = 0

        # Queued when emitted from a non-GUI thread, direct when already on it.
        self._sigAudio.connect(self._applyAudio)
        self._sigState.connect(self._applyState)
        self._sigStatus.connect(self._applyStatus)
        self._sigTranscript.connect(self._applyTranscript)
        self._sigLog.connect(self._applyLog)
        self._sigReady.connect(self._applyReady)
        self._sigSpeaking.connect(self._applySpeaking)
        self._sigTermOpen.connect(self._applyTermOpen)
        self._sigTermOut.connect(self._applyTermOut)
        self._sigTermBusy.connect(self._applyTermBusy)
        self._sigSpokenChunk.connect(self._applySpokenChunk)
        # `paste` and `clip drop` are terminal commands, and terminal commands
        # run on a worker thread. The chip is a QML binding, so the notification
        # has to arrive on the GUI thread like every other one here.
        self._sigClip.connect(self.clipChanged)
        self._sigVault.connect(self._applyVault)
        self._sigClosing.connect(self.setClosing)

        bus.subscribe(TERMINAL_TOGGLE,
                      lambda e: self._sigTermOpen.emit(bool(e.payload.get("open", True))))
        bus.subscribe(TERMINAL_OUTPUT,
                      lambda e: self._sigTermOut.emit(e.payload.get("text", ""),
                                                      e.payload.get("kind", "out")))
        bus.subscribe(ENGINE_READY, lambda e: self._sigReady.emit())
        bus.subscribe(AUDIO_LEVEL, self._onAudio)
        bus.subscribe(STATE_CHANGED, self._onState)
        bus.subscribe(STATUS_WORD, self._onStatus)
        bus.subscribe(TRANSCRIPT_FINAL, self._onTranscript)
        bus.subscribe(LOG_LINE, self._onLog)
        # Speaking is tracked from the audio itself rather than from the state
        # machine. During a generated answer the engine stays in THINKING while
        # the first clauses are already playing, so keying the visuals off state
        # alone would show "WORKING" over a voice that is mid-sentence.
        bus.subscribe(SPEAK_START, lambda e: self._sigSpeaking.emit(True))
        bus.subscribe(SPEAK_END, lambda e: self._sigSpeaking.emit(False))
        # Mirror speech into the terminal as it is spoken, one synthesis unit at
        # a time. At night the pane is the primary channel, and watching the words
        # land in step with the voice is most of what makes it feel like one
        # conversation rather than two.
        bus.subscribe(SPEAK_CHUNK,
                      lambda e: self._sigSpokenChunk.emit(e.payload.get("text", ""),
                                                          bool(e.payload.get("first"))))
        # The calibration switch, and the only place it is ever visible. Not
        # spoken, not on the visualizer, not a question — one dim line in the
        # pane. Going back to HIVE is not worth a line: the operator asked to see
        # the switch, and every switch is followed by a return by construction.
        bus.subscribe(CALIBRATION, self._onCalibration)
        # Consolidation reporting on itself, so the veil can name the file
        # rather than saying "saving" and hoping.
        bus.subscribe(SESSION_SAVED,
                      lambda e: self._sigVault.emit(e.payload.get("stage", ""),
                                                    e.payload.get("name", ""),
                                                    bool(e.payload.get("verified"))))

    def set_quit_callback(self, fn) -> None:
        self._quit_cb = fn

    # -- bus handlers (any thread) ------------------------------------------
    def _onAudio(self, evt) -> None:
        spec = evt.payload.get("spectrum") or []
        self._sigAudio.emit(float(evt.payload.get("rms", 0.0)), spec,
                            evt.payload.get("source", "mic"))

    def _onState(self, evt) -> None:
        self._sigState.emit(evt.payload.get("state", "dormant"))

    def _onStatus(self, evt) -> None:
        self._sigStatus.emit(evt.payload.get("word", ""))

    def _onTranscript(self, evt) -> None:
        self._sigTranscript.emit(evt.payload.get("text", ""))

    def _onLog(self, evt) -> None:
        self._sigLog.emit(evt.payload.get("text", ""), evt.payload.get("kind", "info"))

    def _onCalibration(self, evt) -> None:
        key = evt.payload.get("key", "hive")
        label = evt.payload.get("label", "HIVE")
        if key == "hive":
            return          # the return to base is silent
        self._sigTermOut.emit(f"calibration switched to {label}", "calib")

    # -- GUI-thread appliers -------------------------------------------------
    @Slot(float, list, str)
    def _applyAudio(self, rms: float, spec: list, source: str) -> None:
        if spec:
            n = len(spec)
            if n == SHADER_BANDS:
                self._spectrum = [float(x) for x in spec]
            else:
                # Decimate by max, not by sampling: peaks are what the eye reads,
                # and point-sampling a 128-band spectrum drops half the transients.
                step = n / SHADER_BANDS
                out = []
                for i in range(SHADER_BANDS):
                    lo = int(i * step)
                    hi = max(lo + 1, int((i + 1) * step))
                    out.append(float(max(spec[lo:hi])))
                self._spectrum = out
            self.spectrumChanged.emit()
        # Perceptual lift; raw RMS of speech sits low and looks lifeless.
        lvl = min(1.0, (rms ** 0.5) * 3.4)
        if abs(lvl - self._level) > 0.004:
            self._level = lvl
            self.levelChanged.emit()

    @Slot(str)
    def _applyState(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.stateChanged.emit()
            if state != State.THINKING.value and self._statusWord:
                self._statusWord = ""
                self.statusWordChanged.emit()

    @Slot(str)
    def _applyStatus(self, word: str) -> None:
        if word != self._statusWord:
            self._statusWord = word
            self.statusWordChanged.emit()

    @Slot(str)
    def _applyTranscript(self, text: str) -> None:
        self._transcript = text
        self.transcriptChanged.emit()

    @Slot(str, str)
    def _applyLog(self, text: str, kind: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._log.append(f"{stamp}  {text}")
        self.logChanged.emit()

    @Slot()
    def _applyReady(self) -> None:
        if not self._ready:
            self._ready = True
            self.readyChanged.emit()

    @Slot(bool)
    def _applySpeaking(self, speaking: bool) -> None:
        if speaking != self._speaking:
            self._speaking = speaking
            self.speakingChanged.emit()
        if speaking and self._statusWord:
            self._statusWord = ""       # the word gives way to the voice
            self.statusWordChanged.emit()
        if not speaking:
            # The utterance ended, so the next chunk starts a new line rather
            # than growing the last one.
            self._spoken_open = False

    # -- QML properties ------------------------------------------------------
    @Property(list, notify=spectrumChanged)
    def spectrum(self) -> list:
        return self._spectrum

    @Property(float, notify=levelChanged)
    def level(self) -> float:
        return self._level

    @Property(str, notify=stateChanged)
    def state(self) -> str:
        return self._state

    @Property(str, notify=statusWordChanged)
    def statusWord(self) -> str:
        return self._statusWord

    @Property(str, notify=transcriptChanged)
    def transcript(self) -> str:
        return self._transcript

    @Property(list, notify=logChanged)
    def logLines(self) -> list:
        return list(self._log)

    @Property(bool, notify=readyChanged)
    def ready(self) -> bool:
        return self._ready

    @Property(bool, notify=speakingChanged)
    def speaking(self) -> bool:
        return self._speaking

    # -- terminal ------------------------------------------------------------
    @Slot(str)
    def draft(self, text: str) -> None:
        """Every keystroke in the input row, forwarded to the speculator.

        On the GUI thread and called per character, so it does nothing but hand
        the string over — all the debouncing and the work happens on the
        speculator's own worker.
        """
        if self._draft_cb is not None:
            try:
                self._draft_cb(text or "")
            except Exception:
                pass

    def set_terminal(self, execute, vocabulary=None, on_draft=None) -> None:
        """Hand the bridge the terminal's command executor and its vocabulary.

        `vocabulary` is a callable returning the words worth completing — command
        names and whatever folders are granted right now. It is called per
        keystroke, so it has to stay cheap; Terminal.vocabulary() is a list build
        over a dict that is never large.
        """
        self._term_exec = execute
        self._vocab_fn = vocabulary
        self._draft_cb = on_draft
        self._term_model.append("HIVE terminal. Type `help` for commands.", "sys")

    @Slot(bool)
    def _applyTermOpen(self, open_: bool) -> None:
        if open_ != self._term_open:
            self._term_open = open_
            self.terminalOpenChanged.emit()

    @Slot(str, str)
    def _applyTermOut(self, text: str, kind: str) -> None:
        self._term_model.append(text, kind)
        self.terminalLinesChanged.emit()

    @Slot(str, bool)
    def _applySpokenChunk(self, text: str, first: bool) -> None:
        """One synthesis unit, mirrored into the pane as it goes to the speakers."""
        text = (text or "").strip()
        if not text or not self._term_open:
            return
        if first or not self._spoken_open:
            self._term_model.append(f"HIVE: {text}", "speech")
            self._spoken_open = True
            self._spoken_count += 1
        else:
            self._term_model.extend_last(" " + text)
        self.terminalLinesChanged.emit()

    @Slot(bool)
    def _applyTermBusy(self, busy: bool) -> None:
        if busy != self._term_busy:
            self._term_busy = busy
            self.terminalBusyChanged.emit()

    @Property(bool, notify=terminalOpenChanged)
    def terminalOpen(self) -> bool:
        return self._term_open

    @Property(bool, notify=terminalBusyChanged)
    def terminalBusy(self) -> bool:
        return self._term_busy

    @Property(QObject, constant=True)
    def terminalModel(self) -> QObject:
        return self._term_model

    # NOT constant. It was, and QML cached the empty list from the first read —
    # which quietly disabled arrow-key history for the whole session.
    @Property(list, notify=historyChanged)
    def commandHistory(self) -> list:
        return list(self._history)

    @Slot(bool)
    def setTerminalOpen(self, open_: bool) -> None:
        self._applyTermOpen(open_)

    @Slot(str, result=str)
    def completion(self, text: str) -> str:
        """The greyed-out suffix shown ahead of the caret. '' when there is none.

        History first, then vocabulary — the same order a shell does it in, and
        the one that matches what you are usually reaching for: the long `grant`
        line you typed ten minutes ago, not the word `grants`.
        """
        if not text or not text.strip() or text.endswith(" "):
            return ""
        for past in reversed(self._history):
            if past.startswith(text) and len(past) > len(text):
                return past[len(text):]
        head, _sep, tail = text.rpartition(" ")
        if not tail:
            return ""
        words = self._vocab_fn() if self._vocab_fn is not None else []
        # Shortest first: completing to the nearest word is predictable, and Tab
        # can always be pressed again once more has been typed.
        for word in sorted(set(words), key=lambda w: (len(w), w)):
            if word.startswith(tail) and len(word) > len(tail):
                return word[len(tail):]
        return ""

    @Slot(str)
    def submitCommand(self, text: str) -> None:
        """Run a terminal line off the GUI thread.

        A typed request can run a full HIVE turn — model, skills, speech — which
        takes seconds. Executing it inline would freeze the window and stop the
        visualizer mid-frame.
        """
        text = (text or "").strip()
        if not text or self._term_exec is None:
            return
        self._history.append(text)
        self.historyChanged.emit()
        self._term_model.append(f"> {text}", "in")
        self.terminalLinesChanged.emit()
        self._sigTermBusy.emit(True)
        spoken_before = self._spoken_count

        def work() -> None:
            try:
                out = self._term_exec(text)
            except Exception as exc:
                out = f"error: {type(exc).__name__}: {exc}"
            if out == "\x00clear":
                self._term_model.clear()
                self._sigTermOut.emit("", "out")
            elif out:
                # A conversational line was already mirrored into the pane clause
                # by clause while HIVE said it. Printing the executor's return
                # value on top of that would show the same answer twice.
                mirrored = (self._spoken_count > spoken_before
                            and out.startswith("HIVE:"))
                if not mirrored:
                    self._sigTermOut.emit(out, "out")
            self._sigTermBusy.emit(False)
            # Cheap and unconditional: any line could have been `paste` or
            # `clip drop`, and a chip that disagrees with what is held is worse
            # than no chip at all.
            self._sigClip.emit()

        threading.Thread(target=work, name="hive-term", daemon=True).start()

    # -- clipboard -----------------------------------------------------------
    # Every one of these runs on the GUI thread, called straight from a key
    # handler, so they read and write QClipboard directly and none of them may
    # block. The one thing that could — reasoning over what was pasted — is not
    # here: pasting only *holds* the block, and the turn happens when Enter is
    # pressed like any other.
    def set_clipboard(self, clipboard, clippings, copy_fn=None) -> None:
        self._clipboard = clipboard
        self._clippings = clippings
        self._copy_fn = copy_fn          # Terminal.copy_out — one implementation

    def _receipt(self, text: str) -> None:
        """Say what just happened, in the pane, quietly.

        A copy that succeeds looks exactly like a copy that failed until you go
        somewhere else and paste. One dim line removes the whole question.
        """
        if text:
            self._applyTermOut(text, "sys")

    @Property(int, notify=clipChanged)
    def clipCount(self) -> int:
        return len(self._clippings) if self._clippings is not None else 0

    @Property(str, notify=clipChanged)
    def clipChip(self) -> str:
        return self._clippings.chip() if self._clippings is not None else ""

    @Slot(result=str)
    def paste(self) -> str:
        """Ctrl+V. Returns what should go into the line — '' if it was held.

        The split is the whole design. A path, a URL, a hostname, a hash: those
        are pasted *into* a request and belong in the line. A block of prose is
        the request's subject, and cramming it into a one-line prompt is how it
        ends up as a user turn nobody can ask a second question about.
        """
        if self._clipboard is None:
            return ""
        text = self._clipboard.read()
        if not text.strip():
            return ""
        if self._clippings is None or not self._clippings.is_block(text):
            return text
        clip = self._clippings.hold(text)
        if clip is None:
            return ""
        self.clipChanged.emit()
        if not self._term_open:
            self._applyTermOpen(True)    # holding something invisibly is worse
        self._receipt(f"held {clip.words} words from the clipboard — {clip.label}\n"
                      f"  ask about it in plain language, or `clip drop` to let go")
        return ""

    @Slot(result=str)
    def pasteLiteral(self) -> str:
        """Ctrl+Shift+V. Into the line verbatim, however big it is.

        The escape hatch from the rule above, and it has to exist: the rule is a
        guess about intent, and a guess you cannot overrule is a bug.
        """
        return self._clipboard.read() if self._clipboard is not None else ""

    @Slot(result=str)
    def pastePrimary(self) -> str:
        """Middle-click. The X11/Wayland primary selection, same rules."""
        if self._clipboard is None:
            return ""
        text = self._clipboard.read(selection=True)
        if not text.strip():
            return ""
        if self._clippings is None or not self._clippings.is_block(text):
            return text
        clip = self._clippings.hold(text, source="selection")
        if clip is None:
            return ""
        self.clipChanged.emit()
        self._receipt(f"held {clip.words} words from the selection — {clip.label}")
        return ""

    @Slot()
    def dropClip(self) -> None:
        if self._clippings is None:
            return
        self._receipt(self._clippings.drop())
        self.clipChanged.emit()

    @Slot(int, int)
    def copyRows(self, first: int, last: int) -> None:
        """Whatever is highlighted in the pane."""
        if self._clipboard is None:
            return
        text = self._term_model.text_range(first, last)
        if not text.strip():
            return
        self._clipboard.set(text)
        rows = abs(last - first) + 1
        self._receipt(f"copied {rows} line(s) — {len(text.split())} words.")

    @Slot()
    def copyAnswer(self) -> None:
        """Ctrl+Shift+C with nothing highlighted: HIVE's last answer, whole.

        Straight to the terminal's own implementation rather than reading the
        pane, so the keystroke and `copy` cannot drift apart — and so this gets
        the answer as it was said rather than as it was drawn.
        """
        if self._copy_fn is None:
            return
        # Same rule as submitting into a closed pane: a receipt nobody can see
        # is indistinguishable from the keystroke having been dropped, and "did
        # that copy?" is the exact question this whole path exists to answer.
        if not self._term_open:
            self._applyTermOpen(True)
        try:
            self._receipt(self._copy_fn("last"))
        except Exception as exc:
            self._receipt(f"copy failed: {exc}")

    @Slot(int, result=list)
    def blockAt(self, row: int) -> list:
        lo, hi = self._term_model.block_at(row)
        return [lo, hi]

    # -- actions invoked from QML -------------------------------------------
    @Slot()
    def interrupt(self) -> None:
        """Clicking the visualizer, or Escape, stops HIVE mid-sentence."""
        self.bus.publish("ui.interrupt")

    @Slot()
    def requestQuit(self) -> None:
        if self._quit_cb:
            self._quit_cb()

    # -- shutdown ------------------------------------------------------------
    @Slot(str, str, bool)
    def _applyVault(self, stage: str, name: str, verified: bool) -> None:
        self._vault_stage = stage if verified else "failed"
        self._vault_name = name
        self.vaultChanged.emit()

    @Slot(bool)
    def setClosing(self, closing: bool) -> None:
        if closing != self._closing:
            self._closing = closing
            self.closingChanged.emit()

    def request_closing(self, closing: bool = True) -> None:
        """Raise the veil from any thread.

        Not QTimer.singleShot: a single-shot timer created on a thread with no
        event loop never fires, and `poweroff` arrives here from a worker — so
        the veil would have been missing from the one case it was written for.
        A queued signal is delivered on the thread that owns this object.
        """
        self._sigClosing.emit(closing)

    @Property(bool, notify=closingChanged)
    def closing(self) -> bool:
        return self._closing

    @Property(str, notify=vaultChanged)
    def vaultLine(self) -> str:
        """What the closing veil says under the heading. Never a guess."""
        if self._vault_stage == "summary":
            return f"saved · {self._vault_name}"
        if self._vault_stage == "transcript":
            return f"transcript safe · {self._vault_name}\nwriting the summary…"
        if self._vault_stage == "failed":
            return "THE VAULT WRITE FAILED — check .logs/hive.log"
        return "writing the session to the vault…"
