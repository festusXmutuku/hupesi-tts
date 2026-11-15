"""Desktop app save-file settings: save folder, whether to show Save As, and auto-open after saving.

User feedback: having to choose the folder every time (the dialog always opens in Downloads)
and having to locate the file manually after saving is inconvenient. This only works in the
desktop app (PyWebView): a browser cannot let a web page pick a folder or open a file on the machine.

Configuration lives in ``%LOCALAPPDATA%//HupesiTTS/\save-settings.json``:
``{"folder": str | null, "ask": bool, "open_after": bool}``. If ``folder`` is null or no longer
exists, Downloads is used.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {"folder": None, "ask": True, "open_after": False}


def settings_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "save-settings.json"


def default_folder() -> Path:
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.home()


def load() -> Dict[str, Any]:
    out = dict(DEFAULTS)
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    if isinstance(data.get("folder"), str) and data["folder"].strip():
        out["folder"] = data["folder"]
    for key in ("ask", "open_after"):
        if isinstance(data.get(key), bool):
            out[key] = data[key]
    return out


def save(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and write the configuration (atomic write). The folder must exist."""
    clean = dict(DEFAULTS)
    folder = settings.get("folder")
    if folder:
        p = Path(str(folder)).expanduser()
        if not p.is_dir():
            raise ValueError(f"Folder not found: {folder}")
        clean["folder"] = str(p.resolve())
    for key in ("ask", "open_after"):
        if key in settings:
            clean[key] = bool(settings[key])
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump(clean, f, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return clean


def effective_folder(settings: Optional[Dict[str, Any]] = None) -> Path:
    """Actual save folder: the selected folder if it still exists (USB removed, folder deleted, etc.),
    otherwise Downloads."""
    folder = (settings or load()).get("folder")
    if folder and Path(folder).is_dir():
        return Path(folder)
    return default_folder()


def unique_path(folder: Path, filename: str) -> Path:
    """Path that does not yet exist: "name.wav" -> "name (1).wav"... When saving directly without
    prompting, we must not overwrite a file the user already has."""
    name = Path(filename).name or "cit-voice.wav"
    target = folder / name
    stem, suffix = target.stem, target.suffix
    n = 1
    while target.exists():
        target = folder / f"{stem} ({n}){suffix}"
        n += 1
    return target


def open_path(path: Path, *, reveal: bool = False) -> None:
    """Open the file with the default program, or (``reveal``) open its containing folder and select it."""
    path = Path(path)
    if sys.platform == "win32":
        if reveal:
            # Must be a manually composed command string, not a list: list2cmdline would wrap
            # `/select,C:\...\name.mp3` in a single pair of quotes when it contains spaces,
            # causing Explorer to misread it and open the Documents folder instead.
            subprocess.Popen(f'explorer /select,"{path}"')  # noqa: S603
        else:
            os.startfile(str(path))  # noqa: S606 - open the file the user just saved
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)] if reveal else ["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent if reveal else path)])
