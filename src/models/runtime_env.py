from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_model_runtime_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "pml_model_cache"
    pytensor_cache = cache_root / "pytensor"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    for path in (pytensor_cache, matplotlib_cache, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)

    # Keep PyTensor/Matplotlib caches writable without overriding settings.
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    _set_pytensor_flag("base_compiledir", str(pytensor_cache))


def _set_pytensor_flag(name: str, value: str) -> None:
    existing = os.environ.get("PYTENSOR_FLAGS", "")
    if f"{name}=" in existing:
        return
    flag = f"{name}={value}"
    os.environ["PYTENSOR_FLAGS"] = f"{existing},{flag}" if existing else flag
