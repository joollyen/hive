#!/usr/bin/env bash
# Launch HIVE. This is what HIVE.desktop invokes.
#
# Responsibilities beyond starting Python:
#   - make sure a user-level Ollama is running (HIVE is useless without it)
#   - keep everything inside this directory; nothing is installed system-wide
#
# CUDA library paths are handled in Python (hive/_cuda.py) rather than here, so
# HIVE behaves identically whether it is launched from a shell or from the GNOME
# app grid, which does not source a shell profile.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$HERE/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "HIVE: no virtualenv found. Run ./install.sh first." >&2
    exit 1
fi

# --- Ollama ---------------------------------------------------------------
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/bin/ollama}"
[[ -x "$OLLAMA_BIN" ]] || OLLAMA_BIN="$(command -v ollama || true)"

if ! curl -sf --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    if [[ -n "$OLLAMA_BIN" && -x "$OLLAMA_BIN" ]]; then
        echo "HIVE: starting Ollama..."
        mkdir -p "$HERE/.logs"
        nohup "$OLLAMA_BIN" serve >"$HERE/.logs/ollama-serve.log" 2>&1 &
        for _ in $(seq 1 40); do
            curl -sf --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
            sleep 0.5
        done
    else
        echo "HIVE: Ollama not found — conversation will be disabled." >&2
    fi
fi

# Qt on Wayland. Falls back to xcb automatically if the compositor refuses.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland;xcb}"
# Silence ONNX Runtime's per-session ScatterND advisory; it is expected here.
export ORT_LOGGING_LEVEL="${ORT_LOGGING_LEVEL:-3}"

exec "$PY" -m hive "$@"
