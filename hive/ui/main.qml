import QtQuick
import QtQuick.Window
import QtQuick.Controls

Window {
    id: root
    width: appWidth
    height: appHeight
    visible: true
    color: "#03060d"
    title: "HIVE"
    flags: Qt.Window | Qt.FramelessWindowHint

    // Injected from Python before load.
    property int appWidth: 520
    property int appHeight: 720

    // Closing HIVE ends the session, and that has to stay true now that a second
    // window exists. Qt quits on the *last* window closing, so with HIVE-Q open
    // an Alt+F4 here would leave the engine running behind a query editor with
    // no visualizer and no consolidation on the way out. Routed through
    // requestQuit, which is the same path the × and Ctrl+Q take.
    onClosing: function (close) {
        close.accepted = false
        bridge.requestQuit()
    }

    readonly property string state_: bridge.state
    // Speaking comes from the audio path, not the state machine — see bridge.py.
    // It also suppresses thinking and dormant, since a voice that is mid-sentence
    // is neither of those no matter what the engine state says.
    readonly property bool isSpeaking: bridge.speaking || state_ === "speaking"
    readonly property bool isDormant: state_ === "dormant" && !isSpeaking
    readonly property bool isThinking: state_ === "thinking" && !isSpeaking
    readonly property bool isAlert: state_ === "alert"
    readonly property bool isConfirming: state_ === "confirming" && !isSpeaking

    readonly property color cyan: "#4fb3ff"
    readonly property color dim: "#2a4a72"
    readonly property color hot: "#e8f6ff"
    readonly property color red: "#ff3b48"
    readonly property color edge: isAlert ? red : "#2f7f8c"

    readonly property string stateLabel:
          isAlert ? "VETO"
        : isConfirming ? "CONFIRM"
        : isSpeaking ? "SPEAKING"
        : isThinking ? "WORKING"
        : state_ === "listening" ? "LISTENING"
        : state_ === "transcribing" ? "PARSING"
        : state_ === "closing" ? "CLOSING"
        : "STANDBY"

    // Eased state scalars. Snapping these produces a UI that feels like a status
    // light; easing them produces one that feels like something reacting.
    property real mThinking: isThinking ? 1 : 0
    property real mSpeaking: isSpeaking ? 1 : 0
    property real mDormant: isDormant ? 1 : 0
    property real mAlert: isAlert ? 1 : 0
    Behavior on mThinking { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
    Behavior on mSpeaking { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }
    Behavior on mDormant  { NumberAnimation { duration: 520; easing.type: Easing.OutCubic } }
    Behavior on mAlert    { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }

    property real smoothLevel: 0
    Behavior on smoothLevel { NumberAnimation { duration: 55 } }
    Connections {
        target: bridge
        function onLevelChanged() { root.smoothLevel = bridge.level }
    }

    property real clock: 0
    Timer {
        interval: 16; running: true; repeat: true
        onTriggered: root.clock += 0.016
    }

    // ---------------------------------------------------------------- core
    ShaderEffect {
        id: core
        anchors.fill: parent
        fragmentShader: shaderUrl          // set from Python

        property real uTime: root.clock
        property real uLevel: root.smoothLevel
        property real uAlert: root.mAlert
        property real uSpeaking: root.mSpeaking
        property real uThinking: root.mThinking
        property real uDormant: root.mDormant
        property real uAspect: width / Math.max(1, height)

        property var s: bridge.spectrum
        function v4(i) {
            var a = s || []
            return Qt.vector4d(a[i] || 0, a[i+1] || 0, a[i+2] || 0, a[i+3] || 0)
        }
        property vector4d uS0:  v4(0)
        property vector4d uS1:  v4(4)
        property vector4d uS2:  v4(8)
        property vector4d uS3:  v4(12)
        property vector4d uS4:  v4(16)
        property vector4d uS5:  v4(20)
        property vector4d uS6:  v4(24)
        property vector4d uS7:  v4(28)
        property vector4d uS8:  v4(32)
        property vector4d uS9:  v4(36)
        property vector4d uS10: v4(40)
        property vector4d uS11: v4(44)
        property vector4d uS12: v4(48)
        property vector4d uS13: v4(52)
        property vector4d uS14: v4(56)
        property vector4d uS15: v4(60)
    }

    // -------------------------------------------------------------- readout
    // v2 carried a wordmark, corner brackets, a measurement ruler, legend dots,
    // a dashed connector and a telemetry panel. All of it is gone.
    //
    // The furniture was borrowed from the reference images, where it belongs to
    // a diagram *of* something. Here it sat on top of the subject and competed
    // with it, and a name painted across your own face is a logo on a mirror.
    // What is left is one line: the word for whatever HIVE is doing, and under
    // it what it last heard. Everything else the face already says.
    Item {
        id: header
        anchors { left: parent.left; right: parent.right; top: parent.top }
        height: 108

        Text {
            id: readout
            anchors {
                left: parent.left; leftMargin: 40
                right: parent.right; rightMargin: 40
                top: parent.top; topMargin: 28
            }
            horizontalAlignment: Text.AlignHCenter
            elide: Text.ElideRight
            // The thinking word bank when there is one, the plain state otherwise.
            // Fading between them rather than cutting is what makes a long task
            // read as sustained effort instead of a flickering label.
            text: bridge.statusWord.length ? bridge.statusWord : root.stateLabel
            color: root.isAlert ? root.red : root.cyan
            opacity: bridge.statusWord.length ? 0.90
                   : root.isDormant ? 0.14 : 0.38
            font {
                family: "monospace"
                pixelSize: bridge.statusWord.length ? 15 : 9
                letterSpacing: bridge.statusWord.length ? 1.3 : 3.4
                italic: bridge.statusWord.length > 0
            }
            Behavior on opacity { NumberAnimation { duration: 380 } }

            SequentialAnimation on scale {
                running: bridge.statusWord.length > 0; loops: Animation.Infinite
                NumberAnimation { to: 1.015; duration: 900; easing.type: Easing.InOutSine }
                NumberAnimation { to: 1.0;   duration: 900; easing.type: Easing.InOutSine }
            }
        }

        Text {
            id: heard
            visible: !bridge.statusWord.length && bridge.transcript.length > 0
            anchors {
                left: parent.left; leftMargin: 44
                right: parent.right; rightMargin: 44
                top: readout.bottom; topMargin: 12
            }
            horizontalAlignment: Text.AlignHCenter
            text: "“" + bridge.transcript + "”"
            color: root.hot
            opacity: 0.34
            elide: Text.ElideRight
            font { family: "monospace"; pixelSize: 11; letterSpacing: 0.4; italic: true }
        }
    }

    // -------------------------------------------------------------- footer
    Rectangle {
        id: footer
        anchors { left: parent.left; right: parent.right; bottom: termInputBar.top }
        height: 118
        // The terminal occupies the same space; showing both stacks two
        // scrolling readouts on top of each other.
        opacity: bridge.terminalOpen ? 0 : 1
        visible: opacity > 0.01
        Behavior on opacity { NumberAnimation { duration: 200 } }
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#0003060d" }
            GradientStop { position: 0.42; color: "#cc03060d" }
            GradientStop { position: 1.0; color: "#f203060d" }
        }

        Column {
            anchors {
                left: parent.left; leftMargin: 26
                right: parent.right; rightMargin: 26
                bottom: parent.bottom; bottomMargin: 14
            }
            spacing: 2

            Repeater {
                model: bridge.logLines
                delegate: Text {
                    width: parent.width
                    text: modelData
                    color: root.cyan
                    elide: Text.ElideRight
                    // Older lines fade upward — the readout in ref 2 and ref 3.
                    opacity: 0.10 + 0.34 * (index / Math.max(1, bridge.logLines.length - 1))
                    font { family: "monospace"; pixelSize: 9; letterSpacing: 0.6 }
                }
            }
        }
    }

    // ------------------------------------------------------- HIVE terminal
    // Typed channel alongside the voice one. Opened by saying "HIVE, open your
    // terminal", or Ctrl+` . Grants are made here and never by voice, so a
    // misheard folder name can't reach a file operation.
    // The input row is NOT part of this pane — it lives at the bottom of the
    // window permanently, below. Typing is often faster than speaking, and an
    // input you have to summon first is one you stop reaching for.
    Rectangle {
        id: terminal
        anchors { left: parent.left; right: parent.right; bottom: termInputBar.top }
        height: bridge.terminalOpen ? Math.round(root.height * 0.56) : 0
        visible: height > 2
        clip: true
        // Transparent enough to see the face working behind it, which is the
        // point at night — but readability is not negotiable, so the panel is
        // light and a gradient scrim sits under the text instead. Glass, not
        // paint.
        color: "transparent"
        border { width: bridge.terminalOpen ? 1 : 0; color: root.edge }

        Rectangle {   // scrim: lighter at the top, denser under the input row
            anchors.fill: parent
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#8c060d16" }
                GradientStop { position: 0.55; color: "#b2050b13" }
                GradientStop { position: 1.0; color: "#d904080f" }
            }
        }

        // A slow sheen crossing the panel, so the glass reads as a surface
        // rather than a flat hole cut in the window.
        Rectangle {
            anchors.fill: parent
            opacity: 0.05
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: Math.max(0, (root.clock * 0.05) % 1.4 - 0.2); color: "#0022e0d0" }
                GradientStop { position: Math.max(0.001, (root.clock * 0.05) % 1.4 - 0.1); color: "#5522e0d0" }
                GradientStop { position: Math.min(1, (root.clock * 0.05) % 1.4); color: "#0022e0d0" }
            }
        }
        Behavior on height {
            NumberAnimation { duration: 240; easing.type: Easing.OutCubic }
        }

        // Title strip, minus the title. "HIVE TERMINAL" written across a pane
        // you deliberately opened is the same redundancy as the wordmark.
        Item {
            id: termBar
            anchors { left: parent.left; right: parent.right; top: parent.top }
            height: 22
            Text {
                anchors { right: parent.right; rightMargin: 12; verticalCenter: parent.verticalCenter }
                text: bridge.terminalBusy ? "WORKING" : "READY"
                color: bridge.terminalBusy ? "#ffb347" : root.cyan
                opacity: 0.42
                font { family: "monospace"; pixelSize: 8; letterSpacing: 1.8 }
                Behavior on opacity { NumberAnimation { duration: 200 } }
            }
            Rectangle {
                anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
                height: 1; color: root.edge; opacity: 0.30
            }
        }

        ListView {
            id: termView
            anchors {
                left: parent.left; right: parent.right
                top: termBar.bottom; bottom: parent.bottom
                leftMargin: 14; rightMargin: 14; topMargin: 8; bottomMargin: 8
            }
            clip: true
            // An incremental model — see TerminalModel in bridge.py. The old list
            // property rebuilt every delegate on every appended line, which is
            // what made a busy pane feel like it was fighting you.
            model: bridge.terminalModel
            cacheBuffer: 600
            boundsBehavior: Flickable.StopAtBounds

            // -- selection ------------------------------------------------
            // Whole rows, not characters. A per-delegate TextEdit would give
            // character selection inside one line and nothing across two, and
            // an answer here is a dozen wrapped rows — so the selection that
            // works within a line is precisely the one nobody wants. Row ranges
            // are held by the view because the delegates that make up a range
            // mostly do not exist: the list is virtualised, and the rows above
            // the fold have been recycled.
            property int selFrom: -1
            property int selTo: -1
            readonly property bool hasSel: selFrom >= 0 && selTo >= 0
            function selLo() { return Math.min(selFrom, selTo) }
            function selHi() { return Math.max(selFrom, selTo) }
            function clearSel() { selFrom = -1; selTo = -1 }
            function selectAll() { selFrom = 0; selTo = count - 1 }
            function copySel() {
                if (!hasSel) return false
                bridge.copyRows(selLo(), selHi())
                return true
            }

            // Stick to the bottom the way a terminal should — but glide there
            // rather than jumping, and let go the moment you scroll up yourself.
            property bool stick: true
            readonly property bool atBottom: contentY >= contentHeight - height - 6

            function stickToEnd() {
                var target = Math.max(0, contentHeight - height)
                if (Math.abs(contentY - target) < 0.5) return
                scrollHome.stop()
                scrollHome.from = contentY
                scrollHome.to = target
                scrollHome.start()
            }
            onContentHeightChanged: if (stick) stickToEnd()
            onHeightChanged: if (stick) stickToEnd()
            onMovementStarted: stick = false
            onMovementEnded: stick = atBottom

            NumberAnimation {
                id: scrollHome
                target: termView; property: "contentY"
                duration: 170; easing.type: Easing.OutCubic
            }

            delegate: Item {
                id: rowItem
                width: termView.width
                // Explicit height is required: a wrapped Text keeps the delegate
                // at one line otherwise, and every multi-line entry draws on top
                // of the one after it.
                height: row.implicitHeight

                readonly property bool picked: termView.hasSel
                                               && index >= termView.selLo()
                                               && index <= termView.selHi()

                Rectangle {   // the selection band, bled past the text margins
                    anchors.fill: parent
                    anchors.leftMargin: -7
                    anchors.rightMargin: -7
                    color: "#2f7f8c"
                    opacity: rowItem.picked ? 0.26 : 0
                    Behavior on opacity { NumberAnimation { duration: 90 } }
                }

                Text {
                    id: row
                    width: parent.width
                    text: line
                    wrapMode: Text.WrapAtWordBoundaryOrAnywhere
                    textFormat: Text.PlainText
                    font { family: "monospace"; pixelSize: 11 }
                    color: kind === "in" ? "#eafcff"
                         : kind === "speech" ? "#49e3d8"
                         : kind === "calib" ? "#8a7fd4"
                         : kind === "sys" ? "#5f93b4"
                         : line.startsWith("  ok") ? "#5ff2b8"
                         : line.startsWith("error") || line.startsWith("  --") ? "#ff7b72"
                         : "#a8d8e4"
                    // The calibration line is deliberately the quietest thing in
                    // the pane. It is a receipt, not an announcement.
                    readonly property real base: kind === "in" ? 1.0
                                               : kind === "calib" ? 0.55 : 0.86
                    font.italic: kind === "calib"

                    // Only rows arriving at the bottom animate in. A delegate
                    // that is merely being recycled back into view must not
                    // replay the entrance — that is what made scrolling the
                    // backlog flicker.
                    opacity: 0
                    x: 0
                    Component.onCompleted: {
                        if (index >= termView.count - 2) {
                            lineIn.start()
                        } else {
                            opacity = row.base
                        }
                    }
                    ParallelAnimation {
                        id: lineIn
                        NumberAnimation {
                            target: row; property: "opacity"
                            from: 0; to: row.base; duration: 150; easing.type: Easing.OutQuad
                        }
                        NumberAnimation {
                            target: row; property: "x"
                            from: 7; to: 0; duration: 170; easing.type: Easing.OutCubic
                        }
                    }
                }
            }
            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
        }

        // The pane is read with the mouse and copied with the keyboard, so this
        // sits over the list and takes the drag. It costs drag-to-scroll, which
        // is the right trade twice over: the wheel and the scrollbar still work,
        // and dragging inside a log that is scrolling itself as HIVE talks was
        // never a way to read it anyway.
        //
        // Wheel events are not accepted by a MouseArea, so they fall through to
        // the Flickable underneath untouched. The right margin clears the
        // scrollbar rather than swallowing it.
        MouseArea {
            id: termSelect
            anchors { fill: termView; rightMargin: 14 }
            enabled: bridge.terminalOpen
            acceptedButtons: Qt.LeftButton
            cursorShape: Qt.IBeamCursor
            preventStealing: true

            property int anchorRow: -1
            property bool dragged: false

            function rowAt(mx, my) {
                return termView.indexAt(mx + termView.contentX, my + termView.contentY)
            }

            onPressed: function (mouse) {
                anchorRow = rowAt(mouse.x, mouse.y)
                dragged = false
                termView.clearSel()
            }
            onPositionChanged: function (mouse) {
                if (!pressed || anchorRow < 0) return
                var i = rowAt(mouse.x, mouse.y)
                if (i < 0) return
                dragged = true
                termView.selFrom = anchorRow
                termView.selTo = i
                // Selecting suspends the follow-the-tail behaviour. Having the
                // pane scroll out from under a selection while HIVE keeps
                // talking is the one thing that would make this unusable.
                termView.stick = false
            }
            onReleased: {
                // A click that never moved is a click, not a selection of one
                // line. It clears — which is also how you dismiss a selection.
                if (!dragged) termView.clearSel()
            }
            // Double-click takes the whole exchange: the line you typed and
            // everything HIVE said back. Ctrl+Shift+C then has the answer.
            onDoubleClicked: function (mouse) {
                var i = rowAt(mouse.x, mouse.y)
                if (i < 0) return
                var span = bridge.blockAt(i)
                termView.selFrom = span[0]
                termView.selTo = span[1]
                dragged = true
                termView.stick = false
            }
        }
    }

    // ------------------------------------------------------- input, always on
    // Pinned to the bottom of the window whether the pane is open or not.
    // Typing is often quicker than saying it — exact paths, folder names, a
    // command you want to be certain was heard — and an input you have to open
    // first is one you stop reaching for. The pane above it is just the output
    // half, and it comes up on its own when there is something to show.
    Rectangle {
        id: termInputBar
        anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
        // Grows with the request. Everything above anchors to this bar's top, so
        // the pane and the log readout give way as the input takes more room.
        height: Math.max(38, inputClip.height + 25) + clipChip.height
        Behavior on height { NumberAnimation { duration: 90; easing.type: Easing.OutCubic } }
        gradient: Gradient {
            GradientStop { position: 0.0; color: bridge.terminalOpen ? "#e604080f" : "#b003060d" }
            GradientStop { position: 1.0; color: "#f502050b" }
        }
        Rectangle {   // hairline above, so the strip reads as a surface
            anchors { left: parent.left; right: parent.right; top: parent.top }
            height: 1
            color: root.edge
            opacity: bridge.terminalOpen ? 0.30 : 0.16
            Behavior on opacity { NumberAnimation { duration: 200 } }
        }

        // What is held from the clipboard, and the only reason holding it is
        // safe. A clipping goes to the model with every turn until it is
        // dropped, so it is invisible state that steers answers — and invisible
        // state that steers answers is a bug unless you can see it. One line,
        // and an × that lets go of it.
        Item {
            id: clipChip
            anchors { left: parent.left; leftMargin: 14
                      right: parent.right; rightMargin: 14
                      top: parent.top }
            height: bridge.clipCount > 0 ? 18 : 0
            visible: height > 1
            clip: true
            Behavior on height { NumberAnimation { duration: 140; easing.type: Easing.OutCubic } }

            Text {
                anchors { left: parent.left; verticalCenter: parent.verticalCenter }
                width: parent.width - 40
                text: bridge.clipChip
                color: "#49e3d8"
                opacity: 0.70
                elide: Text.ElideRight
                font { family: "monospace"; pixelSize: 9; letterSpacing: 1.1 }
            }
            Text {
                id: clipDrop
                anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                text: dropArea.containsMouse ? "drop ×" : "×"
                color: dropArea.containsMouse ? root.red : "#49e3d8"
                opacity: dropArea.containsMouse ? 0.95 : 0.55
                font { family: "monospace"; pixelSize: 9; letterSpacing: 1.1 }
                MouseArea {
                    id: dropArea
                    anchors { fill: parent; margins: -6 }
                    hoverEnabled: true
                    onClicked: bridge.dropClip()
                }
            }
        }

        Text {
            id: termPrompt
            text: "$"
            color: root.cyan
            opacity: termInput.activeFocus ? 0.95 : 0.45
            anchors { left: parent.left; leftMargin: 14
                      top: clipChip.bottom; topMargin: 12 }
            font { family: "monospace"; pixelSize: 12; bold: true }
            Behavior on opacity { NumberAnimation { duration: 180 } }
        }

        // A viewport, so a long request wraps and grows the bar upward instead
        // of scrolling sideways out of view. Capped at a third of the window;
        // past that the view follows the last line rather than growing further.
        Item {
            id: inputClip
            anchors {
                left: termPrompt.right; leftMargin: 8
                right: parent.right; rightMargin: 14
                top: clipChip.bottom; topMargin: 10
            }
            height: Math.min(root.height * 0.30,
                             Math.max(17, termInput.implicitHeight))
            clip: true

            TextEdit {
                id: termInput
                width: parent.width
                // Keep the caret's line in view once the text outgrows the cap.
                y: Math.min(0, inputClip.height - implicitHeight)
                color: "#eafcff"
                selectionColor: "#1d6f78"
                selectByMouse: true
                font { family: "monospace"; pixelSize: 12 }
                textFormat: TextEdit.PlainText
                // A TextInput cannot wrap — it is single-line by construction and
                // scrolls horizontally instead, which pushes the start of a long
                // request off the left edge exactly when you want to reread it
                // before submitting. TextEdit wraps; Return has to be intercepted
                // below, since it now means "newline" by default rather than
                // "submit".
                wrapMode: TextEdit.WrapAtWordBoundaryOrAnywhere
                // Always focusable, so a keystroke lands without opening anything
                // first. The visualizer has no other use for the keyboard.
                focus: true
                persistentSelection: false
                property int histIndex: -1

                // Inline completion, the way a shell does it: history first, then
                // command and grant names. Typing is where a terminal either
                // feels quick or doesn't, and finishing the word for you is worth
                // more than any amount of animation.
                property string ghost: ""
                function refreshGhost() { ghost = bridge.completion(text) }
                function acceptGhost() {
                    if (!ghost.length) return false
                    text = text + ghost
                    cursorPosition = text.length
                    ghost = ""
                    return true
                }
                function submit() {
                    if (text.trim().length === 0) return
                    // Bring the output half up on its own. Submitting into a
                    // closed pane and getting nothing back is indistinguishable
                    // from the keystroke having been dropped.
                    if (!bridge.terminalOpen) bridge.setTerminalOpen(true)
                    bridge.submitCommand(text)
                    histIndex = -1
                    ghost = ""
                    text = ""
                }
                // HIVE reads the line as it is typed — see cognition/speculate.py.
                // By the time Enter is pressed the acknowledgement is already
                // rendered and the heavy model is already loading.
                onTextChanged: {
                    refreshGhost()
                    bridge.draft(text)
                }

                // Caret: a solid block that stops blinking while you type, so
                // the pane feels responsive rather than merely animated.
                cursorDelegate: Rectangle {
                    width: 7; color: "#49e3d8"
                    opacity: termInput.activeFocus ? 0.9 : 0.3
                    SequentialAnimation on opacity {
                        running: termInput.activeFocus && termInput.text.length === 0
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.15; duration: 620 }
                        NumberAnimation { to: 0.9; duration: 620 }
                    }
                }

                // Return submits; Shift+Return starts a new line by hand, for the
                // rare multi-part request worth laying out before sending.
                Keys.onReturnPressed: function (event) {
                    if (event.modifiers & Qt.ShiftModifier) { event.accepted = false; return }
                    submit(); event.accepted = true
                }
                Keys.onEnterPressed: function (event) {
                    if (event.modifiers & Qt.ShiftModifier) { event.accepted = false; return }
                    submit(); event.accepted = true
                }
                Keys.onTabPressed: function (event) {
                    // Always accepted: a literal tab in a command line is never
                    // what was meant, so it is swallowed even with no completion.
                    acceptGhost(); event.accepted = true
                }
                Keys.onRightPressed: function (event) {
                    // Only when there is nothing to the right of the caret, so
                    // arrowing through what you already typed still works.
                    event.accepted = (cursorPosition === length) && acceptGhost()
                }
                // History on the arrows, but only from the edges of the text —
                // once a request has wrapped onto several lines, up and down have
                // to move the caret between them first.
                Keys.onUpPressed: function (event) {
                    if (cursorRectangle.y > 1) { event.accepted = false; return }
                    var h = bridge.commandHistory
                    if (h.length === 0) { event.accepted = true; return }
                    histIndex = histIndex < 0 ? h.length - 1 : Math.max(0, histIndex - 1)
                    text = h[histIndex]
                    cursorPosition = text.length
                    event.accepted = true
                }
                Keys.onDownPressed: function (event) {
                    if (cursorRectangle.y + cursorRectangle.height < contentHeight - 1) {
                        event.accepted = false; return
                    }
                    if (histIndex < 0) { event.accepted = false; return }
                    histIndex += 1
                    if (histIndex >= bridge.commandHistory.length) { histIndex = -1; text = "" }
                    else {
                        text = bridge.commandHistory[histIndex]
                        cursorPosition = text.length
                    }
                    event.accepted = true
                }
                // Escape means "back out of whatever is in front of me": drop a
                // pane selection, then a half-typed line, then close the pane,
                // then stop HIVE talking. The input holds focus permanently now,
                // so it owns this key and has to cover all four cases.
                Keys.onEscapePressed: {
                    if (termView.hasSel) termView.clearSel()
                    else if (text.length) { text = ""; ghost = "" }
                    else if (bridge.terminalOpen) bridge.setTerminalOpen(false)
                    else bridge.interrupt()
                }

                // Paste, three ways, all of them explicit.
                //
                // A TextEdit already handles Ctrl+V by itself, and that is the
                // problem rather than the solution: what it does is insert the
                // text into the line, which for a paragraph is exactly the
                // behaviour that made pasting a source into HIVE useless. So the
                // key is intercepted here, the bridge decides whether this is a
                // line or a block, and only a line comes back to be inserted.
                //
                // Shift+Insert and Ctrl+Shift+V are here because a terminal is
                // where these fingers have been trained, and a paste that does
                // nothing teaches you the feature does not exist.
                function pasteInto(s) {
                    if (!s.length) return
                    if (selectedText.length) remove(selectionStart, selectionEnd)
                    insert(cursorPosition, s)
                }

                Keys.onPressed: function (event) {
                    var ctrl = (event.modifiers & Qt.ControlModifier) !== 0
                    var shift = (event.modifiers & Qt.ShiftModifier) !== 0

                    if (shift && event.key === Qt.Key_Insert) {
                        pasteInto(bridge.paste()); event.accepted = true; return
                    }
                    if (!ctrl) return

                    if (event.key === Qt.Key_V) {
                        // Shift forces it into the line verbatim, whatever it is.
                        pasteInto(shift ? bridge.pasteLiteral() : bridge.paste())
                        event.accepted = true
                    } else if (event.key === Qt.Key_C) {
                        // Your own selection in the line wins — it is the nearer
                        // thing and you are looking straight at it. Ctrl+Shift+C
                        // always means the pane, and falls back to the last
                        // answer when nothing is highlighted, which is what it is
                        // reached for nine times in ten.
                        if (shift) {
                            if (!termView.copySel()) bridge.copyAnswer()
                            event.accepted = true
                        } else if (!selectedText.length && termView.hasSel) {
                            termView.copySel(); event.accepted = true
                        }
                    } else if (event.key === Qt.Key_A && shift) {
                        termView.selectAll(); event.accepted = true
                    } else if (event.key === Qt.Key_U) {          // kill line
                        text = ""; event.accepted = true
                    } else if (event.key === Qt.Key_L) {   // clear pane
                        bridge.submitCommand("clear"); event.accepted = true
                    }
                }

                // The completion, drawn at the end of the text wherever wrapping
                // has put it. positionToRectangle rather than measuring the
                // string, because with wrapping the end of the text is no longer
                // its width from the left edge.
                property rect endRect: (termInput.text,
                                        termInput.positionToRectangle(termInput.length))
                Text {
                    x: termInput.endRect.x
                    y: termInput.endRect.y
                    visible: termInput.ghost.length > 0
                    text: termInput.ghost
                    color: "#3d7fa6"; opacity: 0.55
                    font: termInput.font
                }

                Text {   // placeholder
                    anchors.fill: parent
                    visible: !termInput.text.length
                    text: bridge.clipCount > 0
                          ? "ask about what's held   ·   clip drop   ·   help"
                          : "grant FOLDERNAME   ·   recall TERM   ·   paste   ·   help"
                    color: "#3d6d9e"; opacity: 0.45
                    elide: Text.ElideRight
                    font { family: "monospace"; pixelSize: 12 }
                }
            }

            // Middle-click pastes the primary selection — the other half of the
            // Linux clipboard, filled by highlighting text and never by Ctrl+C.
            // Left and right buttons are deliberately not accepted, so they fall
            // through to the TextEdit's own caret and selection handling.
            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.MiddleButton
                onClicked: {
                    termInput.forceActiveFocus()
                    termInput.pasteInto(bridge.pastePrimary())
                }
            }
        }
    }

    // --------------------------------------------------------- interaction
    MouseArea {
        anchors.fill: parent
        // Sits behind the terminal so clicks in the panel reach it.
        z: -1
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        onPressed: function (mouse) {
            if (mouse.button === Qt.RightButton) { bridge.interrupt() }
            else { root.startSystemMove() }     // Wayland-safe window drag
        }
        onDoubleClicked: bridge.interrupt()
    }

    Shortcut {
        sequences: ["Ctrl+`", "Ctrl+~", "F12"]
        onActivated: bridge.setTerminalOpen(!bridge.terminalOpen)
    }

    // HIVE-Q. A second window on the same engine — see ui/app.py for why it is
    // a window rather than a second program.
    Shortcut {
        sequences: ["Ctrl+Shift+Q", "F9"]
        onActivated: queryBridge.setOpen(!queryBridge.open)
    }

    // Quit is a Shortcut rather than a Keys handler on a focus-holding Item,
    // because the input row now holds focus permanently and a key handler that
    // never receives keys is just a comment.
    Shortcut {
        sequence: "Ctrl+Q"
        onActivated: bridge.requestQuit()
    }

    // Close affordance — frameless windows have no titlebar to close from.
    Rectangle {
        anchors { right: parent.right; rightMargin: 14; top: parent.top; topMargin: 14 }
        width: 16; height: 16; radius: 8; color: "transparent"
        border { width: 1; color: closeArea.containsMouse ? root.red : root.dim }
        opacity: closeArea.containsMouse ? 1 : 0.22
        Behavior on opacity { NumberAnimation { duration: 160 } }
        Text {
            anchors.centerIn: parent; text: "×"
            color: closeArea.containsMouse ? root.red : root.dim
            font.pixelSize: 12
        }
        MouseArea {
            id: closeArea
            anchors.fill: parent; hoverEnabled: true
            onClicked: bridge.requestQuit()
        }
    }

    // Closing veil. The mirror image of the boot veil, and it exists for a
    // failure rather than for looks.
    //
    // Consolidation takes between nine and fifty-seven seconds on this box, and
    // for six versions the window sat there through all of it saying STANDBY
    // with a farewell already spoken. Nothing on screen distinguished "writing
    // your conversation to Obsidian" from "hung". Clicking × again — the
    // reasonable thing to do — used to kill the write, and .logs/hive.log has
    // seven sessions that ended exactly that way.
    //
    // So: say what is happening, name the file when it lands, and do not offer
    // the × while it is happening.
    Rectangle {
        anchors.fill: parent
        color: "#03060d"
        opacity: bridge.closing ? 0.94 : 0
        visible: opacity > 0.01
        z: 50
        Behavior on opacity { NumberAnimation { duration: 420; easing.type: Easing.InOutQuad } }

        // Swallows clicks so the window drag and the interrupt handler behind
        // the veil cannot be reached while HIVE is writing.
        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons }

        Column {
            anchors.centerIn: parent
            width: parent.width - 80
            spacing: 14

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "CONSOLIDATING"
                color: root.cyan
                opacity: 0.55
                font { family: "monospace"; pixelSize: 10; letterSpacing: 4 }
                SequentialAnimation on opacity {
                    running: bridge.closing && !bridge.vaultLine.startsWith("saved")
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.18; duration: 900 }
                    NumberAnimation { to: 0.55; duration: 900 }
                }
            }
            Text {
                width: parent.width
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                text: bridge.vaultLine
                color: bridge.vaultLine.indexOf("FAILED") >= 0 ? root.red
                     : bridge.vaultLine.startsWith("saved") ? "#5ff2b8"
                     : "#a8d8e4"
                opacity: 0.80
                font { family: "monospace"; pixelSize: 11; letterSpacing: 0.4 }
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "don't close the window — this is the session being written"
                color: root.dim
                opacity: bridge.vaultLine.startsWith("saved") ? 0 : 0.55
                font { family: "monospace"; pixelSize: 9; letterSpacing: 1.2 }
                Behavior on opacity { NumberAnimation { duration: 300 } }
            }
        }
    }

    // Boot veil: hides the first frames while CUDA kernels compile, so the very
    // first thing seen is composed rather than a stutter.
    Rectangle {
        anchors.fill: parent
        color: "#03060d"
        opacity: bridge.ready ? 0 : 1
        visible: opacity > 0.01
        Behavior on opacity { NumberAnimation { duration: 900; easing.type: Easing.InOutQuad } }
        Text {
            anchors.centerIn: parent
            text: "INITIALISING"
            color: root.cyan; opacity: 0.4
            font { family: "monospace"; pixelSize: 9; letterSpacing: 4 }
            SequentialAnimation on opacity {
                running: !bridge.ready; loops: Animation.Infinite
                NumberAnimation { to: 0.12; duration: 850 }
                NumberAnimation { to: 0.45; duration: 850 }
            }
        }
    }
}
