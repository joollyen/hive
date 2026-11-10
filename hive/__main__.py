"""HIVE entry point.

The CUDA preload must happen before onnxruntime or ctranslate2 are imported
anywhere, which is why it sits above every other import in this file and why the
UI is imported lazily inside main(). Import order here is load-bearing.
"""

from __future__ import annotations

import os
import sys

from . import _cuda

_cuda.preload(verbose="--verbose" in sys.argv or "-v" in sys.argv)


def main() -> int:
    if "--selftest" in sys.argv:
        from .selftest import run
        return run()
    if "--dryrun" in sys.argv:
        from .dryrun import run as dry
        return dry()
    if "--miccheck" in sys.argv:
        from .miccheck import run as mic
        return mic()
    from .ui.app import main as ui_main
    return ui_main(sys.argv)


def _exit(code: int) -> None:
    """Leave without running C++ static destructors.

    ONNX Runtime and CTranslate2 both keep worker threads alive behind static
    objects. Unwinding those at interpreter shutdown races the CUDA context
    teardown and prints `terminate called without an active exception` — a
    cosmetic crash, but one that appears *after* a clean session and looks like
    HIVE fell over on exit. All of HIVE's own cleanup, vault consolidation
    included, has already completed by the time we get here.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _exit(main())
