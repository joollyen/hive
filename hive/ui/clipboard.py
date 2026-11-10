"""The system clipboard, reachable from any thread.

QClipboard may only be touched on the thread that owns the GUI, and almost
nothing in HIVE runs there. The terminal executes each line on its own thread,
the turn loop is on another, and an action plan runs wherever the callosum
happened to call it. All of them have reason to reach the clipboard.

So this is the one object that owns it, and everything else goes through a
signal that Qt delivers on the GUI thread — queued when the caller is elsewhere,
direct when it is already there.

    set()    fire and forget; the caller does not wait
    read()   blocks the calling thread, briefly, for an answer

Reading is deliberately on demand rather than cached. The obvious design is to
watch QClipboard::dataChanged and keep the last value ready — but the signal
fires for every copy in every application on the machine, and answering it means
a synchronous round trip to whichever program owns the selection. That is a
foreign process's latency landing on HIVE's GUI thread, at a moment chosen by
somebody else, for a value nobody has asked for. The visualizer runs at 60fps on
that thread. So: nothing happens until something wants the clipboard.

It also closes an old hole. CLIPBOARD_SET has existed since HIVE learned to open
a site with a question in it — `_plan_launch_ask` publishes the question so it
can be pasted where the URL parameter is ignored, and HIVE says out loud that
the question is on the clipboard. Nothing anywhere subscribed to that event. The
sentence was true about the intent and false about the machine, every time it
was spoken. This subscribes.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QClipboard, QGuiApplication

from ..events import CLIPBOARD_SET, EventBus

# Long enough that a busy source application still answers, short enough that a
# terminal command never looks hung. A clipboard fetch that takes longer than
# this is one the owner is not going to service.
READ_TIMEOUT_S = 2.0


class Clipboard(QObject):
    _sigSet = Signal(str)
    _sigRead = Signal(int)

    def __init__(self, bus: EventBus | None = None, parent=None):
        super().__init__(parent)
        self._clip = QGuiApplication.clipboard()
        self._value = ""
        self._done = threading.Event()
        self._lock = threading.Lock()

        self._sigSet.connect(self._apply_set)
        self._sigRead.connect(self._apply_read)

        if bus is not None:
            bus.subscribe(CLIPBOARD_SET,
                          lambda e: self.set(e.payload.get("text", "")))

    # -- GUI-thread appliers --------------------------------------------------
    @Slot(str)
    def _apply_set(self, text: str) -> None:
        self._clip.setText(text, QClipboard.Clipboard)

    @Slot(int)
    def _apply_read(self, mode: int) -> None:
        try:
            self._value = self._clip.text(QClipboard.Mode(mode)) or ""
        except Exception:
            self._value = ""
        self._done.set()

    # -- callable from anywhere ------------------------------------------------
    @property
    def _on_gui_thread(self) -> bool:
        return QThread.currentThread() is self.thread()

    def set(self, text: str) -> bool:
        """Put text on the clipboard. True if there was anything to put."""
        text = text or ""
        if not text:
            return False
        if self._on_gui_thread:
            self._apply_set(text)
        else:
            self._sigSet.emit(text)
        return True

    def read(self, selection: bool = False) -> str:
        """Whatever is on the clipboard now. '' if nothing, or if it timed out.

        `selection` reads the X11/Wayland primary selection instead — the
        middle-click buffer, which on this desktop is filled by highlighting
        text and never by Ctrl+C. Worth having: half of what gets quoted into
        HIVE was highlighted in a browser a second earlier.
        """
        mode = QClipboard.Selection if selection else QClipboard.Clipboard
        if selection and not self._clip.supportsSelection():
            return ""
        if self._on_gui_thread:
            try:
                return self._clip.text(mode) or ""
            except Exception:
                return ""
        # One reader at a time: the result comes back through a single slot, so
        # two threads racing here would read each other's answer.
        with self._lock:
            self._done.clear()
            self._sigRead.emit(int(mode))
            if not self._done.wait(READ_TIMEOUT_S):
                return ""
            return self._value
