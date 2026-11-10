"""Bridge between the query runner and the HIVE-Q window.

Same contract as ui/bridge.py and for the same reason: the runner lives on
`hive-query`, its tasks execute on `hive-query-task`, results arrive off httpx
threads, and Qt properties may only be touched on the thread that owns the
object. So every inbound event is re-emitted as a private Signal and Qt's
automatic connection queues it onto the GUI thread.

Two models rather than one list property, and the split is the same lesson
TerminalModel already records: the log pane grows continuously while a query
runs, and rebuilding every delegate per appended line makes the pane fight the
editor for the same thread. The task table is small and changes wholesale on
each transition, so it stays a plain list; the log is incremental.

The editor buffer is the source of truth for the queue and it lives here, not in
QML. QML owns the caret and the selection; Python owns the text, because the text
outlives the window — it is saved to the vault's `_state` on the way out and
restored on the next boot, so closing HIVE no more discards an unfinished query
than closing a text editor discards the file.
"""

from __future__ import annotations

from PySide6.QtCore import (Property, QAbstractListModel, QByteArray, QModelIndex,
                            QObject, Qt, Signal, Slot)

from ..events import (QUERY_OUTPUT, QUERY_STATE, QUERY_TASK, QUERY_WINDOW,
                      EventBus)

MAX_LOG = 600


class QueryLogModel(QAbstractListModel):
    """The running pane's backlog. Incremental, for the reasons on TerminalModel."""

    LineRole = Qt.UserRole + 1
    KindRole = Qt.UserRole + 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[tuple[str, str]] = []

    def roleNames(self) -> dict:
        return {self.LineRole: QByteArray(b"line"),
                self.KindRole: QByteArray(b"kind")}

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role: int = LineRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        line, kind = self._rows[index.row()]
        return line if role == self.LineRole else (
            kind if role == self.KindRole else None)

    def append(self, text: str, kind: str = "info") -> None:
        lines = (text or "").split("\n") or [""]
        first = len(self._rows)
        self.beginInsertRows(QModelIndex(), first, first + len(lines) - 1)
        self._rows.extend((line, kind) for line in lines)
        self.endInsertRows()
        excess = len(self._rows) - MAX_LOG
        if excess > 0:
            self.beginRemoveRows(QModelIndex(), 0, excess - 1)
            del self._rows[:excess]
            self.endRemoveRows()

    def clear(self) -> None:
        if not self._rows:
            return
        self.beginResetModel()
        self._rows.clear()
        self.endResetModel()


class QueryBridge(QObject):
    # -- consumed by QML ----------------------------------------------------
    openChanged = Signal()
    stateChanged = Signal()
    tasksChanged = Signal()
    bufferChanged = Signal()

    # -- cross-thread carriers ----------------------------------------------
    _sigOpen = Signal(bool)
    _sigState = Signal(dict)
    _sigTask = Signal(dict)
    _sigLog = Signal(str, str)
    _sigOnline = Signal(bool)

    def __init__(self, bus: EventBus, runner=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self._runner = runner
        self._open = False
        self._state: dict = {"running": False, "paused": False, "index": 0,
                             "total": 0, "current": "", "pause_reason": "",
                             "stopping": False, "elapsed": 0.0}
        self._tasks: list[dict] = []
        self._log = QueryLogModel(self)
        self._online = False

        self._sigOpen.connect(self._applyOpen)
        self._sigState.connect(self._applyState)
        self._sigTask.connect(self._applyTask)
        self._sigLog.connect(self._applyLog)
        self._sigOnline.connect(self._applyOnline)

        bus.subscribe(QUERY_WINDOW,
                      lambda e: self._sigOpen.emit(bool(e.payload.get("open", True))))
        bus.subscribe(QUERY_STATE, lambda e: self._sigState.emit(dict(e.payload)))
        bus.subscribe(QUERY_TASK, lambda e: self._sigTask.emit(dict(e.payload)))
        bus.subscribe(QUERY_OUTPUT,
                      lambda e: self._sigLog.emit(e.payload.get("text", ""),
                                                  e.payload.get("kind", "info")))

    def attach(self, runner) -> None:
        self._runner = runner
        self._refresh_tasks()
        self.bufferChanged.emit()

    # -- GUI-thread appliers -------------------------------------------------
    @Slot(bool)
    def _applyOpen(self, open_: bool) -> None:
        if open_ != self._open:
            self._open = open_
            self.openChanged.emit()
        if open_:
            # Re-announce the buffer whenever the window comes up. The runtime
            # restores last session's query from the vault on the boot thread,
            # which finishes *after* the bridge is attached — so without this the
            # editor binds once against an empty string and the restored list is
            # invisible even though it is loaded. No binding loop: QML assigns
            # the same string it already stored via setBuffer, and TextEdit
            # ignores an assignment equal to its current text.
            self.bufferChanged.emit()
            self._refresh_tasks()

    @Slot(bool)
    def _applyOnline(self, value: bool) -> None:
        if value != self._online:
            self._online = value
            self.stateChanged.emit()

    @Slot(dict)
    def _applyState(self, state: dict) -> None:
        self._state.update(state)
        self.stateChanged.emit()
        self._refresh_tasks()

    @Slot(dict)
    def _applyTask(self, _task: dict) -> None:
        # The payload carries one task, but the window renders all of them and
        # the runner's snapshot is already consistent — re-reading it is both
        # simpler and immune to an event arriving out of order.
        self._refresh_tasks()

    @Slot(str, str)
    def _applyLog(self, text: str, kind: str) -> None:
        self._log.append(text, kind)

    def _refresh_tasks(self) -> None:
        if self._runner is None:
            return
        snap = self._runner.snapshot()
        self._tasks = snap.get("tasks", [])
        self.tasksChanged.emit()

    # -- properties ----------------------------------------------------------
    @Property(bool, notify=openChanged)
    def open(self) -> bool:
        return self._open

    @Property("QVariantMap", notify=stateChanged)
    def state(self) -> dict:
        return dict(self._state)

    @Property(list, notify=tasksChanged)
    def tasks(self) -> list:
        return list(self._tasks)

    @Property(QObject, constant=True)
    def logModel(self) -> QObject:
        return self._log

    @Property(str, notify=bufferChanged)
    def buffer(self) -> str:
        return self._runner.buffer if self._runner is not None else ""

    @Property(bool, notify=stateChanged)
    def online(self) -> bool:
        return self._online

    def set_online(self, value: bool) -> None:
        """Safe to call from any thread — see the note at the top of this file."""
        self._sigOnline.emit(bool(value))

    # -- called from QML -----------------------------------------------------
    @Slot(str)
    def setBuffer(self, text: str) -> None:
        """Every keystroke in the editor. Deliberately cheap — it assigns a
        string and nothing else. Parsing happens when the query is run, not
        while it is being typed."""
        if self._runner is not None:
            self._runner.buffer = text or ""

    @Slot(result=str)
    def run(self) -> str:
        if self._runner is None:
            return "HIVE-Q isn't attached to a runtime."
        self._log.clear()
        ok, msg = self._runner.start()
        if not ok:
            self._log.append(msg, "incomplete")
        return msg

    @Slot(result=str)
    def endQuery(self) -> str:
        """The button form of `end query`. Same call, same override."""
        if self._runner is None:
            return ""
        return self._runner.end("you stopped it from HIVE-Q")

    @Slot()
    def clearLog(self) -> None:
        self._log.clear()

    @Slot(bool)
    def setOpen(self, open_: bool) -> None:
        self._applyOpen(open_)

    @Slot(int, result=str)
    def retryText(self, index: int) -> str:
        """The original line of a task, so an incomplete one can be put back in
        the editor without being retyped."""
        for t in self._tasks:
            if t.get("index") == index:
                return t.get("text", "")
        return ""
