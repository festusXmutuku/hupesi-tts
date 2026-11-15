"""Generation history stored on disk, preserved across app restarts.

Why not store in the browser: the desktop build runs PyWebView, and ``webview.start()``
defaults to ``private_mode=True`` - localStorage/IndexedDB are cleared every time the app closes.

Layout: ``%LOCALAPPDATA%/HupesiTTS/history/``
    index.json        item list (newest first), written atomically
    <name>.wav|.mp3   audio, named like "2026-09-24_20-14-12_Minh Duc_3f2a1b2c"
    <name>.srt        subtitles (if any)
(items saved before readable filenames were introduced use the id as <name>)

NO automatic deletion: generated audio is the user's work. Previously the app silently deleted the
oldest item when exceeding 300 items / 1 GB - but 48 kHz WAV is ~330 MB/hour, so three 1-hour pieces
would fill it, and a single >1 GB piece (~3 hours) was deleted immediately after creation. Now the
History tab shows item count + size + folder (``GET /api/history/info``) and reminds users to clean up
when exceeding ``WARN_BYTES`` or when free disk space drops below ``LOW_DISK_BYTES``.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()
_lock = threading.Lock()

WARN_BYTES = 5 * 1024 ** 3      # history larger than this: suggest cleanup
LOW_DISK_BYTES = 2 * 1024 ** 3  # less free disk space than this: suggest cleanup
AUDIO_CT = {"wav": "audio/wav", "mp3": "audio/mpeg"}
AUDIO_EXT = {v: k for k, v in AUDIO_CT.items()}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")  # prevent path traversal: id is only a hex uuid
_MP3_CHUNK = 30  # seconds per chunk when transcoding WAV -> MP3


def history_dir() -> Path:
    base = os.getenv("LOCALAPPDATA") or str(Path.home())
    p = Path(base) / "HupesiTTS" / "history"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _index_path() -> Path:
    return history_dir() / "index.json"


def _load() -> List[dict]:
    path = _index_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError, TypeError):
        # Corrupted index: do NOT overwrite (it may be recoverable manually); start with an empty list.
        return []


def _save(items: List[dict]) -> None:
    path = _index_path()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".history-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _item_path(item_id: str, fmt: str) -> Path:
    return history_dir() / f"{item_id}.{fmt}"


def _slug(text: str, limit: int = 48) -> str:
    """Readable filename component: keep letters/numbers/spaces and common text marks; strip the rest."""
    text = re.sub(r"[^\w\s\-]+", "", (text or ""), flags=re.UNICODE).strip()
    text = re.sub(r"\s+", " ", text)[:limit].strip().rstrip(".")
    return text or "cit-voice"


def _unique_name(stem: str, fmt: str) -> str:
    name = f"{stem}.{fmt}"
    if not (history_dir() / name).exists():
        return name
    for n in range(1, 1000):
        name = f"{stem} ({n}).{fmt}"
        if not (history_dir() / name).exists():
            return name
    return f"{stem}-{uuid.uuid4().hex[:6]}.{fmt}"


def _storage_name(voice_name: str, fmt: str) -> str:
    """Readable filename when opening the folder: "2026-09-24_20-14-12_Minh Duc_3f2a1b2c"."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"{ts}_{_slug(voice_name)}_{uuid.uuid4().hex[:8]}"
    return _unique_name(stem, fmt)


def _resolve_audio(item: dict) -> Path:
    """Audio path for an item: 'file' field, or legacy <id>.<format>."""
    d = history_dir()
    if item.get("file"):
        p = d / item["file"]
        if p.is_file():
            return p
    return _item_path(str(item.get("id", "")), str(item.get("format", "wav")))


def _files_of(item: dict) -> List[Path]:
    d = history_dir()
    audio = _resolve_audio(item)
    paths = [audio]
    sub = item.get("subtitle_file")
    if sub:
        paths.append(d / sub)
    else:
        paths.append(audio.with_suffix(".srt"))
    return paths


def _delete_files(item: dict) -> None:
    for path in _files_of(item):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass  # file may be locked: leaving a small orphan is better than breaking deletion


def _total_size(items: List[dict]) -> int:
    total = 0
    for item in items:
        for path in _files_of(item):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def _get(item_id: str) -> tuple[List[dict], dict]:
    if not _ID_RE.match(item_id):
        raise HTTPException(status_code=404, detail="This generated item was not found.")
    items = _load()
    for item in items:
        if item.get("id") == item_id:
            return items, item
    raise HTTPException(status_code=404, detail="This generated item was not found.")


class SaveReq(BaseModel):
    audio_base64: str = Field(..., max_length=400_000_000)  # ~300 MB WAV
    format: str = Field("wav", pattern="^(wav|mp3)$")
    text: str = Field("", max_length=5000)
    voice_name: str = ""
    voice_id: str = ""
    duration_sec: float = 0.0
    subtitle_srt: Optional[str] = Field(None, max_length=5_000_000)


@router.get("/api/history")
def list_history():
    """Full history, newest first (without audio - fetch via ``audio_url``)."""
    items = _load()
    return {"items": items, "count": len(items), "total_bytes": _total_size(items)}


@router.get("/api/history/info")
def history_info():
    """Item count, size, folder, and free disk space - for the note line in the History tab.
    ``warn``: suggest that the user clean up."""
    items = _load()
    total = _total_size(items)
    directory = str(history_dir())
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        free = None
    warn = total >= WARN_BYTES or (free is not None and free <= LOW_DISK_BYTES)
    return {
        "count": len(items), "total_bytes": total, "directory": directory,
        "free_bytes": free, "warn": warn, "warn_bytes": WARN_BYTES, "low_disk_bytes": LOW_DISK_BYTES,
    }


@router.post("/api/history/open-folder")
def open_history_folder():
    """Open the history folder in the file manager (on the server machine)."""
    d = history_dir()
    try:
        if sys.platform == "win32":
            os.startfile(d)  # noqa: S606 - the app's own folder
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not open the history folder.") from e
    return {"directory": str(d)}


@router.post("/api/history/{item_id}/reveal")
def reveal_history_item(item_id: str):
    """Open the history folder and select this item's audio file."""
    with _lock:
        _, item = _get(item_id)
        path = _resolve_audio(item)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The audio file for this item has been deleted.")
    try:
        from apps.save_settings import open_path
        open_path(path, reveal=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not open the containing folder.") from e
    return {"path": str(path)}


@router.post("/api/history")
def save_history(req: SaveReq):
    """Save a just-generated audio item (with optional SRT subtitles) to history."""
    import base64
    if req.format not in AUDIO_CT:
        raise HTTPException(status_code=400, detail="Audio format must be wav or mp3.")
    try:
        data = base64.b64decode(req.audio_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio data.")
    if len(data) > 300 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file is too large to save to History.")

    item_id = uuid.uuid4().hex
    fname = _storage_name(req.voice_name, req.format)
    with _lock:
        (history_dir() / fname).write_bytes(data)
        sname = None
        if req.subtitle_srt:
            sname = Path(fname).with_suffix(".srt").name
            # write_bytes: on Windows, write_text would convert \n -> \r\n.
            (history_dir() / sname).write_bytes(req.subtitle_srt.encode("utf-8"))
        item = {
            "id": item_id, "file": fname, "subtitle_file": sname,
            "format": req.format, "text": (req.text or "")[:500],
            "voice_name": req.voice_name, "voice_id": req.voice_id,
            "duration_sec": round(req.duration_sec, 2),
            "size_bytes": len(data), "created_at": datetime.now().isoformat(timespec="seconds"),
            "audio_url": f"/api/history/{item_id}/audio",
            "subtitle_url": f"/api/history/{item_id}/subtitle" if req.subtitle_srt else None,
        }
        items = _load()
        items.insert(0, item)
        _save(items)
    return item


def add_file(audio_path: Path, fmt: str, *, text: str = "", voice_name: str = "", voice_id: str = "",
             duration_sec: float = 0.0, subtitle_srt: Optional[str] = None) -> dict:
    """Write audio (via ``write_audio(path)`` or an existing file) + subtitles and add to the index."""
    item_id = uuid.uuid4().hex
    fname = _storage_name(voice_name, fmt)
    with _lock:
        dest = history_dir() / fname
        os.replace(audio_path, dest)  # same volume as temp_file() -> rename, not copy
        size = dest.stat().st_size
        sname = None
        if subtitle_srt:
            sname = Path(fname).with_suffix(".srt").name
            (history_dir() / sname).write_bytes(subtitle_srt.encode("utf-8"))
        item = {
            "id": item_id, "file": fname, "subtitle_file": sname,
            "format": fmt, "text": (text or "")[:500],
            "voice_name": voice_name, "voice_id": voice_id,
            "duration_sec": round(float(duration_sec), 2),
            "size_bytes": size, "created_at": datetime.now().isoformat(timespec="seconds"),
            "audio_url": f"/api/history/{item_id}/audio",
            "subtitle_url": f"/api/history/{item_id}/subtitle" if subtitle_srt else None,
        }
        items = _load()
        items.insert(0, item)
        _save(items)
    return item


def temp_file(suffix: str = ".wav") -> Path:
    """Temporary file INSIDE the history directory so ``add_file`` can rename it (same volume)
    instead of copying hundreds of MB. Also cleans up stale temp files older than 1 day (e.g. after a crash)."""
    d = history_dir()
    fd, path = tempfile.mkstemp(dir=d, prefix=".gen-", suffix=suffix)
    os.close(fd)
    # Opportunistic cleanup of old temp files.
    try:
        import time
        cutoff = time.time() - 86400
        for p in d.glob(".gen-*"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except OSError:
        pass
    return Path(path)


@router.get("/api/history/{item_id}/audio")
def get_audio(item_id: str):
    with _lock:
        _, item = _get(item_id)
        path = _resolve_audio(item)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The audio file for this item has been deleted.")
    fmt = str(item.get("format", "wav"))
    return FileResponse(str(path), media_type=AUDIO_CT.get(fmt, "application/octet-stream"))


def _wav_to_mp3_file(src: Path, dst: Path) -> bool:
    """Convert WAV -> MP3 in 30-second chunks (RAM usage independent of duration).
    False if this libsndfile build has no MP3 encoder."""
    import numpy as np
    import soundfile as sf
    from apps import ai_label

    with sf.SoundFile(str(src)) as fin:
        block = int(fin.samplerate * _MP3_CHUNK)
        try:
            with ai_label.wav_open(str(dst), "w", samplerate=fin.samplerate, channels=fin.channels) as fout:
                ai_label.label(fout)   # "AI-generated" ID3 tag - must be set before writing
                for chunk in fin.blocks(blocksize=block, dtype="float32", always_2d=True):
                    fout.write(chunk)
        except Exception:
            dst.unlink(missing_ok=True)
            return False
    return True


def export_audio(item_id: str, fmt: str, dest: Path) -> Path:
    """Write a history item's audio to ``dest`` (desktop app WAV/MP3 download button):
    direct copy, or chunked MP3 transcoding - without going through the browser."""
    with _lock:
        _, item = _get(item_id)
        src = _resolve_audio(item)
    if not src.is_file():
        raise HTTPException(status_code=404, detail="The audio file for this item has been deleted.")
    if fmt == str(item.get("format")):
        shutil.copy2(src, dest)
        return dest
    if fmt == "mp3" and str(item.get("format")) == "wav":
        if _wav_to_mp3_file(src, dest):
            return dest
        dest = dest.with_suffix(".wav")   # libsndfile build has no MP3: return WAV
        shutil.copy2(src, dest)
        return dest
    raise HTTPException(status_code=400, detail="This conversion is not supported.")


@router.get("/api/history/{item_id}/audio.mp3")
def get_audio_mp3(item_id: str):
    """Download as MP3 (browser mode; the desktop app uses ``export_audio``)."""
    from fastapi.responses import StreamingResponse
    with _lock:
        _, item = _get(item_id)
        src = _resolve_audio(item)
    if not src.is_file():
        raise HTTPException(status_code=404, detail="The audio file for this item has been deleted.")
    if str(item.get("format")) == "mp3":
        return FileResponse(str(src), media_type="audio/mpeg")
    if str(item.get("format")) != "wav":
        raise HTTPException(status_code=400, detail="This conversion is not supported.")
    # Stream conversion for browser download.
    import io
    import soundfile as sf
    from apps import ai_label
    buf = io.BytesIO()
    try:
        with sf.SoundFile(str(src)) as fin:
            block = int(fin.samplerate * _MP3_CHUNK)
            with ai_label.wav_open(buf, "w", samplerate=fin.samplerate, channels=fin.channels, format="MP3") as fout:
                ai_label.label(fout)
                for chunk in fin.blocks(blocksize=block, dtype="float32", always_2d=True):
                    fout.write(chunk)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not convert to MP3.")
    buf.seek(0)
    return StreamingResponse(iter([buf.read()]), media_type="audio/mpeg")


@router.get("/api/history/{item_id}/subtitle")
def get_subtitle(item_id: str):
    with _lock:
        _, item = _get(item_id)
        sfile = item.get("subtitle_file")
        path = (history_dir() / sfile) if sfile else _resolve_audio(item).with_suffix(".srt")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="This item has no subtitles.")
    return FileResponse(str(path), media_type="text/plain; charset=utf-8", filename=path.name)


@router.delete("/api/history/{item_id}")
def delete_item(item_id: str):
    with _lock:
        items, item = _get(item_id)
        _delete_files(item)
        items = [i for i in items if i.get("id") != item_id]
        _save(items)
    return {"deleted": item_id}


# Compatibility alias
wav_file_to_mp3 = _wav_to_mp3_file
