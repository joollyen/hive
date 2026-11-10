"""CUDA userspace library resolution.

The NVIDIA libraries live in pip packages under site-packages/nvidia/*/lib, not on
the system linker path. Normally you fix that with LD_LIBRARY_PATH, but the dynamic
linker reads that variable at exec time — which means it has to be set by whatever
launches the process. A .desktop file launched from the GNOME overview does not
inherit a shell rc, so relying on it would make HIVE work from a terminal and fail
from the app grid.

Instead we dlopen the libraries ourselves with RTLD_GLOBAL before onnxruntime or
ctranslate2 are ever imported. Once a soname is in the process with global scope,
the provider libraries resolve against it regardless of the search path.

This module MUST be imported before onnxruntime / faster_whisper. Import order in
this package is load-bearing; that is why it is a side-effecting call rather than
something clever.
"""

from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path

# Load order matters: the runtime before the math libraries that link against it,
# and cuDNN last since it depends on both cuBLAS and cudart.
_PRELOAD_ORDER = (
    "libcudart.so.13",
    "libcudart.so.12",
    "libcublas.so.13",
    "libcublasLt.so.13",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libnvrtc.so.13",
    "libcufft.so.12",
    "libcurand.so.10",
    "libcudnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_heuristic.so.9",
)

_loaded: dict[str, bool] = {}
_lib_dirs: list[Path] = []


def _candidate_site_packages() -> list[Path]:
    out: list[Path] = []
    try:
        out.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    usp = getattr(site, "getusersitepackages", None)
    if callable(usp):
        try:
            out.append(Path(usp()))
        except Exception:
            pass
    # Running from a venv without site config still has our own package on sys.path.
    out.append(Path(__file__).resolve().parent.parent / ".venv" / "lib")
    return out


def nvidia_lib_dirs() -> list[Path]:
    """Every site-packages/nvidia/*/lib directory we can find."""
    global _lib_dirs
    if _lib_dirs:
        return _lib_dirs
    seen: set[Path] = set()
    for sp in _candidate_site_packages():
        nv = sp / "nvidia"
        if not nv.is_dir():
            continue
        for child in sorted(nv.iterdir()):
            lib = child / "lib"
            if lib.is_dir() and lib not in seen:
                seen.add(lib)
                _lib_dirs.append(lib)
    return _lib_dirs


def preload(verbose: bool = False) -> dict[str, bool]:
    """dlopen the CUDA stack with global scope. Safe to call more than once.

    Missing libraries are not an error — several are version alternatives of each
    other (cudart 12 *or* 13) and only the ones actually present need to load.
    """
    if _loaded:
        return _loaded

    dirs = nvidia_lib_dirs()
    index: dict[str, Path] = {}
    for d in dirs:
        for so in d.glob("lib*.so.*"):
            index.setdefault(so.name, so)

    for soname in _PRELOAD_ORDER:
        path = index.get(soname)
        if path is None:
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            _loaded[soname] = True
        except OSError as exc:  # pragma: no cover - depends on host CUDA
            _loaded[soname] = False
            if verbose:
                print(f"[cuda] failed {soname}: {exc}", file=sys.stderr)

    # Belt and braces: some CTranslate2 builds shell out to the linker for cuDNN
    # rather than resolving through already-loaded symbols.
    if dirs:
        joined = os.pathsep.join(str(d) for d in dirs)
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if joined not in cur:
            os.environ["LD_LIBRARY_PATH"] = f"{joined}{os.pathsep}{cur}" if cur else joined

    if verbose:
        ok = sum(1 for v in _loaded.values() if v)
        print(f"[cuda] preloaded {ok}/{len(_loaded)} libraries from {len(dirs)} dirs",
              file=sys.stderr)
    return _loaded


def cuda_available() -> bool:
    """True if onnxruntime can actually see a CUDA execution provider."""
    preload()
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False
