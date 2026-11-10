"""Qt application host.

Qt owns the main thread and does nothing but draw. The engine boots on a worker
thread so that model loading — which takes several seconds for Whisper and Kokoro
together — never blocks the first frame. The boot veil in QML covers that window.

Shutdown ordering matters and is easy to get wrong: the runtime must consolidate
to the vault *before* Qt tears down, otherwise closing the window loses the
session. aboutToQuit is connected synchronously for that reason.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
# Imported for its side effect as much as its use: PySide6 only downcasts the
# QML root object to QQuickWindow if the QtQuick module has been loaded first.
from PySide6.QtQuick import QQuickWindow  # noqa: F401

from ..config import load as load_config
from ..events import EventBus
from ..runtime import Runtime
from .bridge import UIBridge
from .clipboard import Clipboard
from .query_bridge import QueryBridge

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv)
    cfg = load_config()
    bus = EventBus()

    QCoreApplication.setApplicationName("HIVE")
    QCoreApplication.setOrganizationName("HIVE")
    app = QGuiApplication(argv)

    runtime = Runtime(cfg, bus)
    bridge = UIBridge(bus)
    query_bridge = QueryBridge(bus, runtime.query)

    # The clipboard belongs to Qt and has to be created after the application
    # object exists. Both halves of the engine want it — the terminal reads it
    # for `paste`, the objective side writes to it when it opens a site with a
    # question — so it is handed to the runtime rather than kept in the UI.
    clipboard = Clipboard(bus)
    runtime.attach_clipboard(clipboard)
    bridge.set_clipboard(clipboard, runtime.clippings, runtime.terminal.copy_out)

    shader = HERE / "shaders" / "hive.frag.qsb"
    if not shader.exists():
        print(f"[hive] missing compiled shader: {shader}\n"
              f"       run: pyside6-qsb -o {shader} {shader.with_suffix('')}",
              file=sys.stderr)
        return 2

    engine = QQmlApplicationEngine()
    ctx = engine.rootContext()
    ctx.setContextProperty("bridge", bridge)
    ctx.setContextProperty("queryBridge", query_bridge)
    ctx.setContextProperty("shaderUrl", QUrl.fromLocalFile(str(shader)))
    ctx.setContextProperty("appWidth", int(cfg.get("ui.width", 520)))
    ctx.setContextProperty("appHeight", int(cfg.get("ui.height", 720)))

    engine.load(QUrl.fromLocalFile(str(HERE / "main.qml")))
    if not engine.rootObjects():
        print("[hive] QML failed to load", file=sys.stderr)
        return 3

    # HIVE-Q, in the same process and therefore on the same engine, the same
    # models and the same microphone. Two processes would each load Whisper and
    # Kokoro — about 4GB of VRAM apiece against an 8GB budget shared with the
    # desktop — and would then fight over the input device. It is a second
    # window rather than a second program for that reason, and `end query`
    # reaching it from HIVE's own terminal is what that buys.
    engine.load(QUrl.fromLocalFile(str(HERE / "query.qml")))
    roots = engine.rootObjects()
    if len(roots) < 2:
        print("[hive] HIVE-Q window failed to load — HIVE will run without it",
              file=sys.stderr)

    window = roots[0]
    opacity = float(cfg.get("ui.opacity", 1.0))
    if opacity < 1.0:
        window.setProperty("opacity", opacity)
    if bool(cfg.get("ui.always_on_top", False)):
        from PySide6.QtCore import Qt
        window.setFlags(window.flags() | Qt.WindowStaysOnTopHint)

    # Right-click / Escape / double-click stop HIVE mid-sentence. Routed through
    # the bus so the UI never holds a reference to the audio stack.
    bus.subscribe("ui.interrupt", lambda e: runtime.speaker.interrupt())
    bridge.set_terminal(runtime.terminal.execute, runtime.terminal.vocabulary,
                        runtime.speculate.draft)
    query_bridge.attach(runtime.query)

    # The online indicator in HIVE-Q's status bar. Polled rather than pushed
    # because connectivity is not an event — nothing tells a process that a
    # cable came out.
    #
    # On a worker thread, not this one. Web.online() is cached for twenty
    # seconds, but the call that finds the cache stale performs a real HEAD
    # request with a four-second connect timeout — and on the GUI thread that is
    # four seconds of frozen visualizer every time the network is slow, which is
    # exactly when it would happen. The bridge marshals the result back.
    def poll_net() -> None:
        threading.Thread(
            target=lambda: query_bridge.set_online(runtime.web.online()),
            name="hive-net-poll", daemon=True).start()

    netpoll = QTimer()
    netpoll.setInterval(5000)
    netpoll.timeout.connect(poll_net)
    netpoll.start()

    # --query opens HIVE-Q on boot, which is what HIVE-Q.desktop launches.
    if "--query" in argv or bool(cfg.get("query.autostart", False)):
        from ..events import QUERY_WINDOW
        QTimer.singleShot(600, lambda: bus.publish(QUERY_WINDOW, open=True))

    # ------------------------------------------------------------- shutdown
    # Closing HIVE writes the session to Obsidian, and that write is the only
    # thing here that matters. What follows is arranged around one rule:
    #
    #     nothing may exit the process while runtime.stop() is still running.
    #
    # The version before this one broke that rule in two places, and
    # .logs/hive.log has seven sessions that were lost to it. Consolidation
    # takes nine to fifty-seven seconds — measured, not estimated — and during
    # that window HIVE looked idle. So:
    #
    #   * The first caller ran runtime.stop() *on whatever thread it arrived
    #     on*. From the × or Ctrl+Q that is the GUI thread, so the window froze
    #     solid for up to a minute, the compositor offered "Force Quit", and
    #     taking it killed the write.
    #   * Every later caller found `stopping` already set, returned instantly,
    #     armed a three-second os._exit, and shot the first one in the back.
    #     Clicking × twice — the obvious thing to do when the first click
    #     appeared to do nothing — was enough to lose the conversation.
    #
    # Now: stop() runs once, on a worker, and every other caller waits for it.
    stopping = threading.Event()
    quitting = threading.Event()

    def shutdown() -> None:
        """Run the engine's teardown exactly once; block until it is finished.

        Safe to call from any thread and any number of times. Later callers do
        not start a second teardown and do not skip past the first one — they
        wait for it, which is the difference between a second × click being
        harmless and it being fatal.
        """
        if stopping.is_set():
            # Generous, because the thing being waited on is the session
            # reaching Obsidian and the measured worst case is under a minute.
            # Bounded, because a teardown that wedges must not leave a window
            # that cannot be closed — and by the time this can expire the
            # transcript has been on disk since the first millisecond of stop().
            if not runtime.stopped.wait(float(cfg.get("session.shutdown_wait_s", 180.0))):
                bus.log("shutdown is still running after the wait; leaving anyway",
                        "warn")
            return
        stopping.set()
        try:
            runtime.stop()      # consolidates to the vault; must finish
        finally:
            runtime.stopped.set()   # belt and braces: stop() sets this too

    def quit_now() -> None:
        """The one way out. Called by the ×, Ctrl+Q, SIGINT and `poweroff`."""
        if quitting.is_set():
            # Already on the way out. Emphatically do not arm a second failsafe
            # — that is precisely the bug this comment block is about.
            return
        quitting.set()

        # The veil goes up first, so a shutdown that takes a minute reads as
        # work in progress rather than as a hang. Through a queued signal, not
        # a single-shot timer: `poweroff` gets here from a worker thread, and a
        # timer created on a thread with no event loop never fires.
        bridge.request_closing(True)

        def finish() -> None:
            try:
                # 1. Teardown, consolidation included. On a worker thread
                #    always, even when the caller was the GUI thread, so the
                #    event loop keeps running and the window keeps drawing
                #    while the vault is written. A frozen window is what
                #    invites a force-quit, and a force-quit is what lost the
                #    sessions.
                shutdown()
            finally:
                # 2. Only now is leaving hard safe. The failsafe exists because
                #    app.quit() called from a worker thread does not always
                #    return — measured on this box — and without it the window
                #    can outlive the engine. Armed *after* the write, never
                #    before it, and in a finally so a teardown that raises
                #    still leaves rather than stranding the window.
                def hard_exit() -> None:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)

                t = threading.Timer(3.0, hard_exit)
                t.daemon = True
                t.start()

                # 3. Ask nicely. If the event loop notices, it exits cleanly.
                app.quit()

        threading.Thread(target=finish, name="hive-shutdown", daemon=True).start()

    bridge.set_quit_callback(quit_now)
    runtime.request_quit = quit_now      # lets a spoken "goodnight" close the window
    # Qt is on its way out for a reason we did not initiate — a session logout,
    # or app.quit() from somewhere unexpected. Consolidate before the event loop
    # unwinds; this blocks aboutToQuit, which is exactly what it is for.
    app.aboutToQuit.connect(shutdown)

    # Ctrl-C in a terminal should behave like closing the window, consolidation
    # included. Qt swallows SIGINT unless the loop is woken periodically.
    #
    # SIGTERM and SIGHUP get the same treatment, which they did not before. They
    # are what a desktop logout and a closing terminal send, and with no handler
    # the default action is to die on the spot — so ending the session by
    # logging out lost it, silently, every time. There is no guarantee the
    # writer finishes before the system escalates to SIGKILL, but the transcript
    # is now the first thing stop() does rather than the last, so the window in
    # which that matters is milliseconds instead of a minute.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, lambda *_: quit_now())
        except (OSError, ValueError, AttributeError):
            pass        # not every signal exists on every platform
    wake = QTimer()
    wake.start(300)
    wake.timeout.connect(lambda: None)

    # --screenshot PATH [SECONDS] — render, grab the window, exit. Used to check
    # the visualizer without a display session, and to compare shader changes.
    if "--screenshot" in argv:
        i = argv.index("--screenshot")
        out = Path(argv[i + 1]) if len(argv) > i + 1 else Path("hive-shot.png")
        delay = float(argv[i + 2]) * 1000 if len(argv) > i + 2 else 4000.0

        def grab() -> None:
            img = window.grabWindow()
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"[hive] screenshot {'saved' if img.save(str(out)) else 'FAILED'}: {out}")
            quit_now()

        # --speak LINE makes HIVE say something just before the grab, so the
        # capture lands mid-utterance and shows the visualizer actually driven by
        # audio rather than idling.
        if "--speak" in argv:
            line = argv[argv.index("--speak") + 1]
            QTimer.singleShot(
                max(0, int(delay) - 1400),
                lambda: threading.Thread(
                    target=runtime.speaker.say, args=(line,),
                    kwargs={"cached": False}, daemon=True).start(),
            )

        QTimer.singleShot(int(delay), grab)

    # --demo-terminal opens the pane and types commands into it. Independent of
    # --screenshot: it is used both to capture the terminal in use and to drive
    # end-to-end tests (including "poweroff") without a human at the keyboard.
    if "--demo-terminal" in argv:
        cmds = [c for c in os.environ.get(
            "HIVE_DEMO_CMDS", "grant voss,grants,structure voss").split(",") if c.strip()]
        start_ms = int(os.environ.get("HIVE_DEMO_DELAY_MS", "16000"))

        def demo() -> None:
            from ..events import TERMINAL_TOGGLE
            bus.publish(TERMINAL_TOGGLE, open=True)
            for line in cmds:
                bridge.submitCommand(line.strip())
                time.sleep(1.4)

        QTimer.singleShot(start_ms,
                          lambda: threading.Thread(target=demo, daemon=True).start())

    def boot() -> None:
        # Without this, any exception during boot dies unseen on a daemon thread
        # and the symptom is a window that renders, reacts to your voice, and
        # never answers — which is indistinguishable from a dozen other faults.
        try:
            runtime.start()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            bus.log(f"BOOT FAILED: {type(exc).__name__}: {exc}", "error")

    threading.Thread(target=boot, name="hive-boot", daemon=True).start()
    return app.exec()
