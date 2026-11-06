"""Registry and runner for the five inpainting algorithms.

Each algorithm lives in its own module under backend/algorithms/ and exposes:
    CONNECTED : bool                      (True once your implementation is plugged in)
    inpaint(image, mask, **kwargs)        -> RGB uint8 array, same shape as image
    load()                                (optional, one-time model loading)

To add or rename an algorithm, edit ALGORITHMS below.
"""
import base64
import importlib
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from utils.image_io import encode_png

log = logging.getLogger("inpainting-lab")


@dataclass(frozen=True)
class AlgorithmInfo:
    key: str
    name: str
    category: str
    description: str
    module: str


ALGORITHMS = [
    AlgorithmInfo("telea", "Telea", "Classical",
                  "Fast marching method: fills the hole from its border inward.",
                  "algorithms.classical.telea"),
    AlgorithmInfo("navier_stokes", "Navier-Stokes", "Classical",
                  "Propagates image structure along isophotes using fluid-dynamics equations.",
                  "algorithms.classical.navier_stokes"),
    AlgorithmInfo("patchmatch", "PatchMatch", "Patch-based",
                  "Copies the best-matching patches from the intact part of the image.",
                  "algorithms.patchmatch.patchmatch"),
    AlgorithmInfo("ksvd", "K-SVD", "Machine learning",
                  "Sparse coding over a learned dictionary of image patches.",
                  "algorithms.ksvd.ksvd"),
    AlgorithmInfo("lama", "LaMa", "Deep learning",
                  "Big-LaMa: a pretrained network with fast Fourier convolutions.",
                  "algorithms.lama.lama"),
]
BY_KEY = {a.key: a for a in ALGORITHMS}

_loaded = {}  # key -> None (ok) or error message
_load_lock = threading.Lock()


def _import(info: AlgorithmInfo):
    return importlib.import_module(info.module)


def _ensure_loaded(info: AlgorithmInfo, module) -> None:
    """Run the optional module.load() exactly once, so heavy models are reused."""
    with _load_lock:
        if info.key not in _loaded:
            try:
                if hasattr(module, "load"):
                    module.load()
                _loaded[info.key] = None
            except Exception as exc:  # noqa: BLE001 - reported to the client
                _loaded[info.key] = f"{type(exc).__name__}: {exc}"
                log.exception("Loading %s failed", info.name)
    if _loaded[info.key]:
        raise RuntimeError(f"{info.name} failed to load: {_loaded[info.key]}")


def list_algorithms() -> list:
    out = []
    for info in ALGORITHMS:
        entry = {"key": info.key, "name": info.name, "category": info.category,
                 "description": info.description, "available": False, "reason": None}
        try:
            module = _import(info)
            entry["available"] = bool(getattr(module, "CONNECTED", False))
            if not entry["available"]:
                entry["reason"] = "Not connected yet"
        except Exception as exc:  # noqa: BLE001
            entry["reason"] = f"Module cannot be imported: {exc}"
        out.append(entry)
    return out


def warmup() -> None:
    """Load models of connected algorithms at server start (errors are kept, not raised)."""
    for info in ALGORITHMS:
        try:
            module = _import(info)
            if getattr(module, "CONNECTED", False):
                _ensure_loaded(info, module)
        except Exception as exc:  # noqa: BLE001
            log.warning("Warm-up of %s skipped: %s", info.name, exc)


def _validate_output(result, image: np.ndarray) -> np.ndarray:
    if not isinstance(result, np.ndarray):
        raise TypeError(f"inpaint() must return a numpy array, got {type(result).__name__}.")
    if result.dtype != np.uint8:
        raise TypeError(f"inpaint() must return uint8, got {result.dtype}.")
    if result.shape != image.shape:
        raise ValueError(f"inpaint() must return shape {image.shape} (H, W, 3 RGB), got {result.shape}.")
    return result


def run_one(key: str, image: np.ndarray, mask: np.ndarray, params: dict) -> dict:
    """Run a single algorithm. Never raises: the outcome is described in the returned dict."""
    info = BY_KEY[key]
    result = {"algorithm": info.key, "name": info.name, "category": info.category,
              "status": "error", "time_ms": None, "message": None, "image_base64": None}
    try:
        module = _import(info)
    except Exception as exc:  # noqa: BLE001
        result["message"] = f"Module cannot be imported: {type(exc).__name__}: {exc}"
        return result
    if not getattr(module, "CONNECTED", False):
        result.update(status="unavailable", message="Not connected yet. See backend/algorithms/ for instructions.")
        return result
    try:
        _ensure_loaded(info, module)
        start = time.perf_counter()
        # Every algorithm gets its own copy of the identical original image and mask.
        output = module.inpaint(image.copy(), mask.copy(), **params.get(key, {}))
        elapsed = (time.perf_counter() - start) * 1000
        output = _validate_output(output, image)
        result.update(status="success", time_ms=round(elapsed, 1),
                      image_base64=base64.b64encode(encode_png(output)).decode("ascii"))
    except Exception as exc:  # noqa: BLE001 - isolated per algorithm
        log.exception("%s failed", info.name)
        result["message"] = f"{type(exc).__name__}: {exc}"
    return result


def run_algorithms(keys: list, image: np.ndarray, mask: np.ndarray, params: dict) -> list:
    return [run_one(k, image, mask, params) for k in keys]
