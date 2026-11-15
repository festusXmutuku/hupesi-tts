"""Save the user's pronunciation dictionary and provide an editing API for the UI.

The file lives at ``%LOCALAPPDATA%/HupesiTTS/pronunciation-dict.json`` (same location as
``server-settings.json``) so it persists across app reinstalls/upgrades. Users can also edit this
file manually: the app reloads it when the file modification time changes, without requiring a restart.
"""
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vieneu_utils.pronunciation import (
    DEFAULT_ENTRIES,
    MAX_ENTRIES,
    PronunciationDict,
    validate_entries,
)


def dict_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "pronunciation-dict.json"


_lock = threading.Lock()
# (path, mtime) of the most recent load; None = not loaded yet. mtime None = file does not exist.
_cache_key: tuple | None = None
_cache: PronunciationDict = PronunciationDict(validate_entries(DEFAULT_ENTRIES))


def _read_file(path: Path) -> PronunciationDict:
    """Read the dictionary file. If the file is corrupted, use defaults and DO NOT overwrite the file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data["entries"] if isinstance(data, dict) else data
        return PronunciationDict(validate_entries(raw))
    except FileNotFoundError:
        return PronunciationDict(validate_entries(DEFAULT_ENTRIES))
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"⚠️ Pronunciation dictionary error ({path}): {e} - temporarily using defaults.")
        return PronunciationDict(validate_entries(DEFAULT_ENTRIES))


def current_dict() -> PronunciationDict:
    """The active dictionary; automatically reloads when the file is modified externally."""
    global _cache_key, _cache
    path = dict_path()
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        key = (str(path), None)
    with _lock:
        if key != _cache_key:
            _cache = _read_file(path)
            _cache_key = key
        return _cache


def apply(text: str) -> str:
    """Replace words in ``text`` according to the active dictionary."""
    return current_dict().apply(text)


def save_entries(raw: list) -> PronunciationDict:
    """Validate and write atomically (temp file + os.replace). Invalid data raises ValueError."""
    global _cache_key, _cache
    entries = validate_entries(raw)
    path = dict_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": [
            {"word": e.word, "reading": e.reading, "match_case": e.match_case}
            for e in entries
        ],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as f:
            temporary = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with _lock:
        _cache = PronunciationDict(entries)
        _cache_key = (str(path), path.stat().st_mtime_ns)
    return _cache


def _as_json(d: PronunciationDict) -> list:
    return [{"word": e.word, "reading": e.reading, "match_case": e.match_case} for e in d.entries]


router = APIRouter()


class EntryModel(BaseModel):
    word: str = Field(max_length=200)
    reading: str = Field(max_length=500)
    match_case: bool = True


class EntriesReq(BaseModel):
    entries: List[EntryModel] = Field(max_length=MAX_ENTRIES)


class PreviewReq(BaseModel):
    text: str = Field(max_length=5000)
    # Unsaved draft from the UI; if omitted, use the saved dictionary.
    entries: List[EntryModel] | None = Field(default=None, max_length=MAX_ENTRIES)


@router.get("/api/pronunciations", tags=["Pronunciation"])
def get_pronunciations():
    """Current pronunciation dictionary + defaults (for the restore button)."""
    return {
        "entries": _as_json(current_dict()),
        "defaults": DEFAULT_ENTRIES,
        "path": str(dict_path()),
    }


@router.put("/api/pronunciations", tags=["Pronunciation"])
def put_pronunciations(req: EntriesReq):
    """Save the entire pronunciation dictionary (replacing the old list)."""
    try:
        d = save_entries([e.model_dump() for e in req.entries])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail="Could not save the pronunciation dictionary. Please check configuration write permissions.",
        ) from e
    return {"entries": _as_json(d), "path": str(dict_path())}


@router.post("/api/pronunciations/preview", tags=["Pronunciation"])
def preview_pronunciations(req: PreviewReq):
    """Preview text after applying the dictionary (works even without saving)."""
    if req.entries is None:
        d = current_dict()
    else:
        try:
            d = PronunciationDict(validate_entries([e.model_dump() for e in req.entries]))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return {"text": d.apply(req.text)}
