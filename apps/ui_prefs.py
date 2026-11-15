"""React UI preferences (light/dark theme, auto-play after generation, playback speed, default voice, etc.)
persisted to disk instead of only localStorage.

Why: the desktop app calls ``webview.start()`` without ``private_mode=False``, and PyWebView defaults
to ``private_mode=True``, so localStorage is cleared every time the app closes. Even with private mode
disabled, localStorage is tied to the origin: if the port changes (8001 busy -> 8002), it is lost.
A file on disk is unaffected by both.

Only keys in ``_FIELDS`` are accepted, each type-checked; unknown keys are ignored. The API address
is not stored (it is needed to reach this endpoint in the first place).

``%LOCALAPPDATA%//HupesiTTS//ui-prefs.json``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends

from apps.remote_access import require_this_machine

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_number(lo: float, hi: float):
    return lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


_FIELDS = {
    "theme": lambda v: v in ("light", "dark"),
    "autoPlayOnGenerate": lambda v: isinstance(v, bool),
    "playbackSpeed": _is_number(0.25, 4.0),
    "silenceDuration": _is_number(0.0, 10.0),
    # Empty = use the model's default voice.
    "defaultVoiceId": lambda v: isinstance(v, str) and len(v) <= 200,
}


def settings_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "ui-prefs.json"


def _clean(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in _FIELDS and _FIELDS[k](v)}


def load() -> Dict[str, Any]:
    try:
        return _clean(json.loads(settings_path().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def update(changes: Any) -> Dict[str, Any]:
    """Merge valid keys into the file (atomic write) and return all preferences."""
    prefs = {**load(), **_clean(changes)}
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".ui-prefs-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return prefs


@router.get("/api/ui/prefs")
def get_prefs() -> Dict[str, Any]:
    return load()


@router.put("/api/ui/prefs", dependencies=[Depends(require_this_machine)])
def put_prefs(changes: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
        return update(changes)
    except OSError as e:
        logger.warning("Could not save UI preferences: %s", e)
        return {**load(), **_clean(changes)}
