#!/usr/bin/env bash
# HIVE installer. Entirely userspace — no sudo, nothing written outside $HOME.
#
# Idempotent: safe to re-run after pulling changes or breaking the venv.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p .logs models/kokoro models/whisper models/vad assets/cache

say() { printf '\033[96m::\033[0m %s\n' "$*"; }
ok()  { printf '\033[92m ✓\033[0m %s\n' "$*"; }
warn(){ printf '\033[93m ! \033[0m%s\n' "$*"; }

# --- 1. Python ------------------------------------------------------------
# The system Python is 3.14, which has no wheels for this ML stack. uv fetches a
# managed 3.12 without touching anything system-wide.
if ! command -v uv >/dev/null 2>&1; then
    say "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x .venv/bin/python ]]; then
    say "creating Python 3.12 environment"
    uv venv --python 3.12 .venv
fi
ok "python $(.venv/bin/python --version | cut -d' ' -f2)"

# --- 2. Dependencies ------------------------------------------------------
# Ordering is load-bearing. See the header of requirements.txt: kokoro-onnx pulls
# CPU onnxruntime, which unpacks over onnxruntime-gpu's core library and silently
# removes CUDA support. Installing the GPU build last makes it win.
say "installing dependencies (a few minutes on a cold cache)"
uv pip install --python .venv/bin/python -r requirements.txt >.logs/pip.log 2>&1
uv pip install --python .venv/bin/python --reinstall onnxruntime-gpu >>.logs/pip.log 2>&1

# nvidia-cudnn-cu12 and -cu13 both write libcudnn*.so.9 into the same directory
# and produce a mixed-version file set. Only cu13 belongs here.
if .venv/bin/python -c "import importlib.metadata as m; m.version('nvidia-cudnn-cu12')" 2>/dev/null; then
    warn "removing conflicting nvidia-cudnn-cu12"
    uv pip uninstall --python .venv/bin/python nvidia-cudnn-cu12 >>.logs/pip.log 2>&1 || true
    uv pip install --python .venv/bin/python --reinstall nvidia-cudnn-cu13 >>.logs/pip.log 2>&1
fi
ok "dependencies installed"

# --- 3. Models ------------------------------------------------------------
fetch() {  # url dest label
    [[ -s "$2" ]] && { ok "$3 (cached)"; return; }
    say "downloading $3"
    curl -fL# "$1" -o "$2" || { warn "failed: $3"; return 1; }
    ok "$3"
}

KB=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
fetch "$KB/kokoro-v1.0.fp16-gpu.onnx" models/kokoro/kokoro-v1.0.fp16-gpu.onnx "Kokoro TTS (GPU)"
fetch "$KB/kokoro-v1.0.onnx"          models/kokoro/kokoro-v1.0.onnx          "Kokoro TTS (CPU fallback)"
fetch "$KB/voices-v1.0.bin"           models/kokoro/voices-v1.0.bin           "Kokoro voices"
fetch "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
      models/vad/silero_vad.onnx "Silero VAD"

say "fetching Whisper (downloads on first load)"
.venv/bin/python - <<'PY' >>.logs/pip.log 2>&1 || warn "whisper prefetch failed"
import sys; sys.path.insert(0, ".")
from hive import _cuda; _cuda.preload()
from faster_whisper import WhisperModel
WhisperModel("small.en", device="cuda", compute_type="int8_float16",
             download_root="models/whisper")
PY
ok "Whisper small.en"

# --- 4. Shader ------------------------------------------------------------
say "compiling visualizer shader"
.venv/bin/pyside6-qsb --glsl "100 es,120,150,330,410,440" --hlsl 50 --msl 12 \
    -o hive/ui/shaders/hive.frag.qsb hive/ui/shaders/hive.frag
ok "shader compiled"

# --- 5. Ollama ------------------------------------------------------------
OLLAMA_BIN="$HOME/.local/bin/ollama"
if [[ ! -x "$OLLAMA_BIN" ]] && ! command -v ollama >/dev/null 2>&1; then
    say "installing Ollama to ~/.local (userspace, no sudo)"
    curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
        -o .logs/ollama.tar.zst
    zstd -d -c .logs/ollama.tar.zst | tar -xf - -C "$HOME/.local"
    rm -f .logs/ollama.tar.zst
fi
[[ -x "$OLLAMA_BIN" ]] || OLLAMA_BIN="$(command -v ollama)"
ok "ollama $("$OLLAMA_BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"

if ! curl -sf --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    nohup "$OLLAMA_BIN" serve >.logs/ollama-serve.log 2>&1 &
    for _ in $(seq 1 40); do
        curl -sf --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

FAST=$(.venv/bin/python -c "import tomllib;print(tomllib.load(open('config/hive.toml','rb'))['llm']['fast_model'])")
HEAVY=$(.venv/bin/python -c "import tomllib;print(tomllib.load(open('config/hive.toml','rb'))['llm']['heavy_model'])")
for m in "$FAST" "$HEAVY"; do
    if "$OLLAMA_BIN" list 2>/dev/null | grep -q "^${m%%:*}.*${m##*:}"; then
        ok "model $m (present)"
    else
        say "pulling $m"
        "$OLLAMA_BIN" pull "$m"
    fi
done

# --- 6. Desktop entry -----------------------------------------------------
chmod +x run.sh install.sh
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
for entry in HIVE.desktop HIVE-Q.desktop; do
    sed -e "s|@HIVE_DIR@|$HERE|g" "$entry" >"$APPS/$entry"
    chmod +x "$APPS/$entry"
done
update-desktop-database "$APPS" 2>/dev/null || true
# Two entries, one program. HIVE-Q launches `run.sh --query`, which is HIVE with
# the query window already open — not a second process. Two would each load
# Whisper and Kokoro and then fight over the microphone.
ok "desktop entries -> $APPS/HIVE.desktop, $APPS/HIVE-Q.desktop"

echo
say "verifying"
.venv/bin/python -m hive --selftest || warn "self test reported problems (see above)"

echo
ok "HIVE installed. Launch it from the app grid, or run ./run.sh"
