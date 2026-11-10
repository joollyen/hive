// HIVE-Q — the query.
//
// Notepad++'s *layout*, not its palette: a tab strip across the top, a
// line-numbered gutter down the left of the editor, and a status bar along the
// bottom. That structure is what makes a list of tasks feel like a document you
// own rather than a form you are filling in, which is the whole reason Jay
// asked for an editor instead of a text field.
//
// The colours stay HIVE's. A white Notepad++ window sitting next to the
// visualizer would read as a different program that happened to be running at
// the same time, and this is meant to read as the same thing, elsewhere.
//
// Four tabs, and they are the states a task can be in rather than open files:
//   QUEUE      the editor. The source of truth. Editable while a query runs.
//   RUNNING    what HIVE is saying, live, as it says it.
//   INCOMPLETE what fell over, why, and the line to paste back.
//   DONE       what landed, and the note it was written to.

import QtQuick
import QtQuick.Window
import QtQuick.Controls

Window {
    id: qroot
    width: 760
    height: 620
    minimumWidth: 480
    minimumHeight: 380
    visible: queryBridge.open
    color: "#04070e"
    title: "HIVE-Q"

    readonly property var st: queryBridge.state
    readonly property bool isRunning: st.running === true
    readonly property bool isPaused: st.paused === true

    readonly property color cyan: "#4fb3ff"
    readonly property color teal: "#49e3d8"
    readonly property color dim: "#2a4a72"
    readonly property color red: "#ff7b72"
    readonly property color amber: "#ffb347"
    readonly property color green: "#5ff2b8"
    readonly property color edge: "#2f7f8c"

    property int tab: 0
    readonly property var tabNames: ["QUEUE", "RUNNING", "INCOMPLETE", "DONE"]

    function countOf(status) {
        var n = 0
        for (var i = 0; i < queryBridge.tasks.length; i++)
            if (queryBridge.tasks[i].status === status) n++
        return n
    }
    function tabBadge(i) {
        if (i === 2) { var a = countOf("incomplete") + countOf("skipped"); return a ? " " + a : "" }
        if (i === 3) { var d = countOf("done"); return d ? " " + d : "" }
        return ""
    }

    // Closing the window must not stop the query. It is background work by
    // construction, and a queue that dies when you tidy your desktop is not one
    // you would trust with anything long. `end query` is the only thing that
    // stops it, which is the point of having one command that does.
    onClosing: function (close) {
        close.accepted = false
        queryBridge.setOpen(false)
    }

    // ------------------------------------------------------------- toolbar
    Rectangle {
        id: toolbar
        anchors { left: parent.left; right: parent.right; top: parent.top }
        height: 42
        color: "#070c16"

        Text {
            id: mark
            anchors { left: parent.left; leftMargin: 16; verticalCenter: parent.verticalCenter }
            text: "HIVE·Q"
            color: qroot.teal
            opacity: 0.75
            font { family: "monospace"; pixelSize: 12; letterSpacing: 2.4; bold: true }
        }

        Row {
            anchors { right: parent.right; rightMargin: 14; verticalCenter: parent.verticalCenter }
            spacing: 8

            // Run and End are one control in two states. Two buttons would mean
            // the wrong one is always the bigger target, and the one you reach
            // for in a hurry is End.
            Rectangle {
                width: 104; height: 24; radius: 3
                color: runArea.containsMouse
                       ? (qroot.isRunning ? "#3a1418" : "#0d2a33") : "transparent"
                border { width: 1; color: qroot.isRunning ? qroot.red : qroot.edge }
                Text {
                    anchors.centerIn: parent
                    text: qroot.isRunning ? "END QUERY" : "▶  RUN"
                    color: qroot.isRunning ? qroot.red : qroot.teal
                    font { family: "monospace"; pixelSize: 10; letterSpacing: 1.6; bold: true }
                }
                MouseArea {
                    id: runArea
                    anchors.fill: parent; hoverEnabled: true
                    onClicked: {
                        if (qroot.isRunning) { queryBridge.endQuery() }
                        else { queryBridge.run(); qroot.tab = 1 }
                    }
                }
            }

            Rectangle {
                width: 66; height: 24; radius: 3
                color: clearArea.containsMouse ? "#0d1a24" : "transparent"
                border { width: 1; color: qroot.dim }
                Text {
                    anchors.centerIn: parent
                    text: "CLEAR"
                    color: qroot.dim
                    font { family: "monospace"; pixelSize: 10; letterSpacing: 1.6 }
                }
                MouseArea {
                    id: clearArea
                    anchors.fill: parent; hoverEnabled: true
                    onClicked: queryBridge.clearLog()
                }
            }
        }

        Rectangle {
            anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
            height: 1; color: qroot.edge; opacity: 0.30
        }
    }

    // ----------------------------------------------------------- tab strip
    Row {
        id: tabs
        anchors { left: parent.left; leftMargin: 10; top: toolbar.bottom; topMargin: 6 }
        spacing: 2

        Repeater {
            model: qroot.tabNames
            delegate: Rectangle {
                required property int index
                required property string modelData
                width: label.implicitWidth + 26
                height: 24
                color: qroot.tab === index ? "#0d1622" : "transparent"
                border {
                    width: 1
                    color: qroot.tab === index ? qroot.edge : "transparent"
                }
                Text {
                    id: label
                    anchors.centerIn: parent
                    text: modelData + qroot.tabBadge(index)
                    color: qroot.tab === index ? qroot.cyan
                         : index === 2 && qroot.countOf("incomplete") > 0 ? qroot.red
                         : qroot.dim
                    opacity: qroot.tab === index ? 1.0 : 0.72
                    font { family: "monospace"; pixelSize: 9; letterSpacing: 1.5 }
                }
                MouseArea { anchors.fill: parent; onClicked: qroot.tab = index }
            }
        }
    }

    // --------------------------------------------------------------- panes
    Item {
        id: body
        anchors {
            left: parent.left; right: parent.right
            top: tabs.bottom; topMargin: 4
            bottom: statusbar.top; bottomMargin: 4
            leftMargin: 10; rightMargin: 10
        }

        // -- QUEUE: the editor ------------------------------------------
        Rectangle {
            anchors.fill: parent
            visible: qroot.tab === 0
            color: "#060b13"
            border { width: 1; color: "#4d2f7f8c" }   // edge at 30%; QQuickPen has no opacity

            // Gutter. Line numbers plus a marker on whichever line is running,
            // which is the one thing a plain text editor cannot tell you and the
            // reason this pane is worth having open while the query works.
            Rectangle {
                id: gutter
                anchors { left: parent.left; top: parent.top; bottom: parent.bottom }
                width: 46
                color: "#040810"
                Rectangle {
                    anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                    width: 1; color: qroot.edge; opacity: 0.25
                }
                Column {
                    y: 10 - editorFlick.contentY
                    width: parent.width
                    Repeater {
                        model: Math.max(1, editor.lineCount)
                        delegate: Item {
                            required property int index
                            width: gutter.width
                            height: editor.contentHeight / Math.max(1, editor.lineCount)
                            Text {
                                anchors {
                                    right: parent.right; rightMargin: 10
                                    verticalCenter: parent.verticalCenter
                                }
                                text: index + 1
                                color: qroot.dim
                                opacity: 0.55
                                font { family: "monospace"; pixelSize: 11 }
                            }
                        }
                    }
                }
            }

            Flickable {
                id: editorFlick
                anchors {
                    left: gutter.right; right: parent.right
                    top: parent.top; bottom: parent.bottom
                    leftMargin: 10; rightMargin: 10; topMargin: 10; bottomMargin: 10
                }
                clip: true
                contentWidth: width
                contentHeight: editor.contentHeight
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                TextEdit {
                    id: editor
                    width: editorFlick.width
                    color: "#dceaf2"
                    selectionColor: "#1d6f78"
                    selectByMouse: true
                    focus: true
                    textFormat: TextEdit.PlainText
                    wrapMode: TextEdit.WrapAtWordBoundaryOrAnywhere
                    font { family: "monospace"; pixelSize: 13 }
                    // Python owns the text; QML owns the caret. The buffer
                    // outlives this window, so it cannot live in a QML property.
                    text: queryBridge.buffer
                    onTextChanged: queryBridge.setBuffer(text)

                    Text {
                        anchors { left: parent.left; top: parent.top }
                        visible: !editor.text.length
                        color: "#2d5f7a"
                        opacity: 0.75
                        font: editor.font
                        text: "one task per line — write them the way you'd say them\n\n" +
                              "hive find me research articles on neural networks\n" +
                              "hive research options on best buy for keyboards like the corsair vanguard\n" +
                              "hive open youtube and type into the search bar 'betterminecraft+ hardcore gameplay'\n\n" +
                              "# lines starting with # are notes, not tasks\n\n" +
                              "ctrl+enter runs it · `end query` in HIVE's terminal stops everything"
                    }
                }
            }
        }

        // -- RUNNING: live output ----------------------------------------
        Rectangle {
            anchors.fill: parent
            visible: qroot.tab === 1
            color: "#060b13"
            border { width: 1; color: "#4d2f7f8c" }   // edge at 30%; QQuickPen has no opacity

            ListView {
                id: logView
                anchors { fill: parent; margins: 10 }
                clip: true
                model: queryBridge.logModel
                cacheBuffer: 800
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                property bool stick: true
                readonly property bool atBottom: contentY >= contentHeight - height - 6
                onContentHeightChanged: if (stick) positionViewAtEnd()
                onMovementStarted: stick = false
                onMovementEnded: stick = atBottom

                delegate: Text {
                    required property string line
                    required property string kind
                    width: logView.width
                    height: implicitHeight
                    text: line
                    wrapMode: Text.WrapAtWordBoundaryOrAnywhere
                    textFormat: Text.PlainText
                    font { family: "monospace"; pixelSize: 11 }
                    color: kind === "task" ? "#eafcff"
                         : kind === "speech" ? qroot.teal
                         : kind === "done" ? qroot.green
                         : kind === "incomplete" ? qroot.red
                         : kind === "pause" ? qroot.amber
                         : kind === "note" ? "#7f8fb4"
                         : "#a8d8e4"
                    opacity: kind === "note" ? 0.6 : kind === "task" ? 1.0 : 0.88
                }
            }

            Text {
                anchors.centerIn: parent
                visible: logView.count === 0
                text: qroot.isRunning ? "working…" : "nothing has run yet"
                color: qroot.dim
                font { family: "monospace"; pixelSize: 11; letterSpacing: 1.4 }
            }
        }

        // -- INCOMPLETE / DONE: task tables ------------------------------
        Repeater {
            model: [2, 3]
            delegate: Rectangle {
                required property int modelData
                anchors.fill: parent
                visible: qroot.tab === modelData
                color: "#060b13"
                border { width: 1; color: "#4d2f7f8c" }   // edge at 30%; QQuickPen has no opacity

                readonly property var rows: {
                    var out = []
                    var all = queryBridge.tasks
                    for (var i = 0; i < all.length; i++) {
                        var s = all[i].status
                        if (modelData === 2 && (s === "incomplete" || s === "skipped")) out.push(all[i])
                        if (modelData === 3 && s === "done") out.push(all[i])
                    }
                    return out
                }

                ListView {
                    anchors { fill: parent; margins: 10 }
                    clip: true
                    model: parent.rows
                    spacing: 8
                    boundsBehavior: Flickable.StopAtBounds
                    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                    delegate: Column {
                        id: taskRow
                        required property var modelData
                        width: ListView.view.width
                        spacing: 2

                        Row {
                            spacing: 8
                            Text {
                                id: statusGlyph
                                text: (taskRow.modelData.status === "done" ? "✓" :
                                       taskRow.modelData.status === "skipped" ? "·" : "✗")
                                      + " " + taskRow.modelData.index + "."
                                color: taskRow.modelData.status === "done" ? qroot.green
                                     : taskRow.modelData.status === "skipped" ? qroot.dim
                                     : qroot.red
                                font { family: "monospace"; pixelSize: 12; bold: true }
                            }
                            Text {
                                width: taskRow.width - statusGlyph.width - 16
                                text: taskRow.modelData.title
                                color: "#dceaf2"
                                elide: Text.ElideRight
                                font { family: "monospace"; pixelSize: 12 }
                            }
                        }
                        Text {
                            visible: taskRow.modelData.reason.length > 0
                            x: 22
                            width: taskRow.width - 30
                            text: taskRow.modelData.reason
                            color: qroot.amber
                            opacity: 0.8
                            wrapMode: Text.WordWrap
                            font { family: "monospace"; pixelSize: 10 }
                        }
                        Text {
                            visible: taskRow.modelData.note.length > 0
                            x: 22
                            text: "query_tasks/" + taskRow.modelData.note
                                  + (taskRow.modelData.results > 0
                                     ? "   ·   " + taskRow.modelData.results + " result(s)" : "")
                                  + "   ·   " + taskRow.modelData.duration + "s"
                            color: "#7f8fb4"
                            opacity: 0.65
                            elide: Text.ElideMiddle
                            width: taskRow.width - 30
                            font { family: "monospace"; pixelSize: 9 }
                        }
                    }
                }

                Text {
                    anchors.centerIn: parent
                    visible: parent.rows.length === 0
                    text: modelData === 2 ? "nothing has failed" : "nothing finished yet"
                    color: qroot.dim
                    font { family: "monospace"; pixelSize: 11; letterSpacing: 1.4 }
                }
            }
        }
    }

    // ----------------------------------------------------------- status bar
    Rectangle {
        id: statusbar
        anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
        height: 24
        color: "#070c16"

        Rectangle {
            anchors { left: parent.left; right: parent.right; top: parent.top }
            height: 1; color: qroot.edge; opacity: 0.30
        }

        // A progress hairline under the bar. The only moving thing in the
        // window when it is minimised to a corner of the screen, which is where
        // it will actually live while a query runs.
        Rectangle {
            anchors { left: parent.left; top: parent.top }
            height: 2
            width: qroot.st.total > 0
                   ? parent.width * (qroot.st.index / qroot.st.total) : 0
            color: qroot.isPaused ? qroot.amber : qroot.teal
            opacity: qroot.isRunning ? 0.55 : 0
            Behavior on width { NumberAnimation { duration: 300; easing.type: Easing.OutCubic } }
            Behavior on opacity { NumberAnimation { duration: 300 } }
        }

        Text {
            anchors { left: parent.left; leftMargin: 14; verticalCenter: parent.verticalCenter }
            color: qroot.isPaused ? qroot.amber : qroot.isRunning ? qroot.teal : qroot.dim
            font { family: "monospace"; pixelSize: 9; letterSpacing: 1.2 }
            text: {
                if (qroot.st.stopping === true) return "ENDING…"
                if (qroot.isPaused) return "HELD — " + qroot.st.pause_reason
                if (qroot.isRunning)
                    return "TASK " + qroot.st.index + " OF " + qroot.st.total
                           + "   ·   " + Math.round(qroot.st.elapsed) + "s"
                           + "   ·   " + qroot.st.current
                return "IDLE   ·   " + editor.lineCount + " LINE(S)"
            }
            elide: Text.ElideRight
            width: parent.width - 190
        }

        // Online is stated rather than assumed. HIVE spent five versions saying
        // it was an offline service and the sentence is now conditional, so the
        // condition is on screen.
        Text {
            anchors { right: parent.right; rightMargin: 14; verticalCenter: parent.verticalCenter }
            text: queryBridge.online ? "● ONLINE" : "○ OFFLINE"
            color: queryBridge.online ? qroot.green : qroot.dim
            opacity: 0.75
            font { family: "monospace"; pixelSize: 9; letterSpacing: 1.2 }
        }
    }

    // ------------------------------------------------------------ shortcuts
    Shortcut {
        sequences: ["Ctrl+Return", "Ctrl+Enter"]
        onActivated: if (!qroot.isRunning) { queryBridge.run(); qroot.tab = 1 }
    }
    Shortcut {
        // The keyboard form of the override, and it is Ctrl+Shift+E rather than
        // Escape on purpose: Escape is what you press to dismiss a window, and
        // stopping a running query must never be something you do by reflex.
        sequence: "Ctrl+Shift+E"
        onActivated: queryBridge.endQuery()
    }
    Shortcut { sequence: "Ctrl+1"; onActivated: qroot.tab = 0 }
    Shortcut { sequence: "Ctrl+2"; onActivated: qroot.tab = 1 }
    Shortcut { sequence: "Ctrl+3"; onActivated: qroot.tab = 2 }
    Shortcut { sequence: "Ctrl+4"; onActivated: qroot.tab = 3 }
    Shortcut { sequence: "Escape"; onActivated: queryBridge.setOpen(false) }
}
