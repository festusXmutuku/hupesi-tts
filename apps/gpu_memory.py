"""VRAM limit for the GPU model (v3 Turbo PyTorch/CUDA).

Why this is needed: the batched engine keeps every captured CUDA graph (one per combination of
batch size x cache length x temperature...), each using ~370 MB of KV cache + memory pool.
Measured on an RTX 3050: loading the model uses 0.6 GB, and synthesizing a few passages with
different temperatures reaches ~2.9 GB (excluding the CUDA context) and never drops. On 4 GB cards,
the Windows driver does not report "out of memory" but silently spills into system RAM ("Sysmem
Fallback") - making the machine extremely slow. Batch size barely changes peak VRAM (batches 1-16 all
use ~1.1-1.5 GB) while batch 16 is 4x faster than batch 1, so this module limits the NUMBER OF GRAPHS,
not the batch size.

Two layers of protection:
- ``max_fused_graphs`` (LRU) computed from the allowed VRAM: when exceeded, the oldest graph is dropped.
- ``torch.cuda.set_per_process_memory_fraction``: PyTorch raises OutOfMemoryError when the limit is hit
  instead of spilling to RAM; ``v3turbo._infer_chunks`` catches that error, clears graphs, and retries
  with a smaller batch.

Configuration (``{"vram_limit_gb": null | number}``, null = automatic) is stored at
``%LOCALAPPDATA%/HupesiTTS/gpu-memory.json``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from apps.remote_access import require_this_machine

logger = logging.getLogger(__name__)

GB = 1 << 30
# Measurements on RTX 3050 (see module docstring). One synthesis with 1 graph needs an additional
# ~0.7 GB beyond the model; each extra graph keeps ~0.37 GB - rounded up for safety.
FIRST_GRAPH_BYTES = int(0.75 * GB)
PER_GRAPH_BYTES = int(0.4 * GB)
MAX_GRAPHS = 8
# Automatic: reserve headroom for Windows, WebView2, browser, etc. sharing the same GPU.
AUTO_RESERVE_TOTAL = 1 * GB      # always reserve at least this much of total VRAM
AUTO_HEADROOM_FREE = GB // 2     # and keep at least this much free VRAM
MIN_LIMIT_GB = 1.5
MAX_LIMIT_GB = 80.0

# web_stream attaches a function that returns the running engine here (avoids circular import).
engine_getter: Callable[[], Any] | None = None
# Result of the most recent application, for GET /api/gpu/memory.
_state: Dict[str, Any] = {}


def settings_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "gpu-memory.json"


def configured_limit_gb() -> Optional[float]:
    try:
        value = json.loads(settings_path().read_text(encoding="utf-8")).get("vram_limit_gb")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and MIN_LIMIT_GB <= value <= MAX_LIMIT_GB:
        return float(value)
    return None


def save_limit_gb(limit_gb: Optional[float]) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump({"vram_limit_gb": limit_gb}, f)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def plan(total: int, free: int, reserved: int, model_bytes: int, limit_gb: Optional[float]) -> Dict[str, int]:
    """Compute the VRAM cap (bytes) and number of graphs to keep. Pure function, does not touch torch.

    ``free`` is the GPU's currently free VRAM; ``reserved`` is the amount PyTorch in this app holds
    (already outside ``free``); ``model_bytes`` is the portion occupied by the model."""
    if limit_gb is None:
        limit = min(total - AUTO_RESERVE_TOTAL, free + reserved - AUTO_HEADROOM_FREE)
    else:
        limit = min(int(limit_gb * GB), total)
    limit = max(int(MIN_LIMIT_GB * GB), limit)
    spare = limit - model_bytes - FIRST_GRAPH_BYTES
    graphs = 1 + max(0, spare // PER_GRAPH_BYTES)
    return {"limit": int(limit), "graphs": int(min(MAX_GRAPHS, graphs))}


def _torch_engine(tts) -> Any:
    """Batch engine of the running GPU model, or None (CPU/ONNX, Nano, Supertonic)."""
    if tts is None or getattr(tts, "backend", None) != "pytorch":
        return None
    get = getattr(tts, "_get_batch_engine", None)
    return get() if get else None


def apply(tts, *, just_loaded: bool = True) -> Optional[Dict[str, Any]]:
    """Apply the limit to a newly loaded model (``just_loaded``) or when the user changes the level
    (possibly mid-inference). Never breaks model loading: errors are only logged."""
    engine = _torch_engine(tts)
    if engine is None:
        _state.clear()
        return None
    try:
        import torch

        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        limit_gb = configured_limit_gb()
        if just_loaded or "model_bytes" not in _state:
            # Measure model footprint when no graphs exist yet. When changing mid-run, reuse this value:
            # at that point memory_allocated includes temporary inference buffers.
            engine.trim_fused(0)
            model_bytes = torch.cuda.memory_allocated(dev)
        else:
            model_bytes = _state["model_bytes"]
        # free + reserved = the portion of the GPU available to this app, whether idle or busy.
        reserved = torch.cuda.memory_reserved(dev)
        free, _ = torch.cuda.mem_get_info(dev)
        p = plan(total, free, reserved, model_bytes, limit_gb)
        engine.max_fused_graphs = p["graphs"]
        # An in-progress graph is still held by local variables; removing it from the cache is safe.
        engine.trim_fused(p["graphs"])
        torch.cuda.set_per_process_memory_fraction(min(1.0, p["limit"] / total), dev)
        _state.clear()
        _state.update(
            gpu_name=torch.cuda.get_device_name(dev), total=total, limit=p["limit"],
            graphs=p["graphs"], limit_gb=limit_gb, model_bytes=model_bytes,
        )
        logger.info(
            f"🎛️ VRAM: limit {p['limit'] / GB:.1f}/{total / GB:.1f} GB "
            f"({'automatic' if limit_gb is None else 'user-set'}), keeping at most {p['graphs']} CUDA graphs"
        )
        return dict(_state)
    except Exception as e:  # noqa: BLE001 - VRAM configuration must not block execution
        logger.warning(f"Could not apply VRAM limit: {e}")
        return None


def _gb(x: Optional[int]) -> Optional[float]:
    return None if x is None else round(x / GB, 2)


def status() -> Dict[str, Any]:
    tts = engine_getter() if engine_getter else None
    engine = _torch_engine(tts)
    out: Dict[str, Any] = {"available": False, "limit_gb": configured_limit_gb(),
                           "min_gb": MIN_LIMIT_GB}
    if engine is None or not _state:
        return out
    used = None
    try:
        import torch
        used = torch.cuda.memory_reserved()
    except Exception:  # noqa: BLE001
        pass
    out.update(
        available=True, gpu_name=_state["gpu_name"], total_gb=_gb(_state["total"]),
        effective_limit_gb=_gb(_state["limit"]), used_gb=_gb(used),
        max_graphs=_state["graphs"], graphs=len(getattr(engine, "_fused", {})),
    )
    return out


router = APIRouter()


class GpuMemorySettings(BaseModel):
    limit_gb: Optional[float] = Field(default=None, description="null = automatic")


@router.get("/api/gpu/memory")
def get_gpu_memory():
    return status()


@router.put("/api/gpu/memory")
def put_gpu_memory(settings: GpuMemorySettings, request: Request):
    require_this_machine(request)
    limit = settings.limit_gb
    if limit is not None and not MIN_LIMIT_GB <= limit <= MAX_LIMIT_GB:
        raise HTTPException(status_code=400, detail=f"VRAM limit must be at least {MIN_LIMIT_GB:g} GB.")
    try:
        save_limit_gb(limit)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not save VRAM settings. Please check configuration write permissions.") from error
    # Apply immediately to the running GPU model - no restart needed.
    apply(engine_getter() if engine_getter else None, just_loaded=False)
    return status()
