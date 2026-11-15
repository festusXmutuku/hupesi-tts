"""Check for new releases on GitHub Releases. Only NOTIFIES; does not download or install automatically.

- New version available: the UI shows a dialog (change group names are taken from `###` headings
  in the release notes). "Later" records `skipped` for that version so the dialog is not shown again
  until the next version; the small notification bar and manual check button remain available.

- When the app opens, the UI calls GET /api/app/update. The server asks GitHub at most once per day
  (results saved in %LOCALAPPDATA%/HupesiTTS/update-check.json), waits up to 5 seconds; if the
  network is unavailable it returns `error` and the UI stays silent.
- This is the only time the app connects to the Internet automatically, so it is OFF by default:
  users enable it in Settings (PUT /api/app/update-check). When disabled, only the "Check for updates"
  button in the Info tab (force=1) queries GitHub.
- Only a single GET request to api.github.com is sent, with no text, audio, or machine information.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from apps.app_version import APP_VERSION
from apps.remote_access import require_this_machine
from apps.runtime_paths import https_context

RELEASES_REPO = "festusxmutukugithub/hupesi-tts"
RELEASES_PAGE = f"https://github.com/{RELEASES_REPO}/releases/latest"
LATEST_API = f"https://api.github.com/repos/{RELEASES_REPO}/releases/latest"
CHECK_INTERVAL_SECONDS = 24 * 3600
TIMEOUT_SECONDS = 5
MAX_HIGHLIGHTS = 6
_MARKDOWN = re.compile(r"[*_`]|\[([^\]]*)\]\([^)]*\)")

_lock = threading.Lock()


def config_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "update-check.json"


def parse_version(text: str) -> tuple[int, ...]:
    """"v1.0.1", "1.0.1", "Hupesi TTS 1.2" -> (1, 0, 1) / (1, 2, 0). No numbers -> ()."""
    numbers = [int(n) for n in re.findall(r"\d+", text or "")[:3]]
    return tuple(numbers + [0] * (3 - len(numbers))) if numbers else ()


def is_newer(latest: str, current: str = APP_VERSION) -> bool:
    new, old = parse_version(latest), parse_version(current)
    return bool(new) and new > old


def _read() -> dict:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(state: dict) -> None:
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            json.dump(state, file)
        os.replace(file.name, path)
    except OSError:
        pass  # Cannot write: the next launch will ask GitHub again, which is fine.


def is_enabled() -> bool:
    return _read().get("enabled") is True


def highlights(body: str) -> list[str]:
    """`### Heading` items from release notes with markdown stripped: the "New version available"
    dialog lists only change group names; details are on the Releases page."""
    out = []
    for line in (body or "").splitlines():
        if line.startswith("### "):
            title = _MARKDOWN.sub(lambda m: m.group(1) or "", line[4:]).strip()
            if title:
                out.append(title[:80])
        if len(out) == MAX_HIGHLIGHTS:
            break
    return out


def fetch_latest() -> dict:
    """{"version", "url", "published_at"} of the latest published release (skipping drafts/prereleases)."""
    request = urllib.request.Request(LATEST_API, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"hupesi-tts/{APP_VERSION}",
    })
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS,
                                context=https_context()) as response:
        body = json.loads(response.read().decode("utf-8"))
    url = body.get("html_url") or ""
    # Only open the correct release repository page; do not trust arbitrary links in the response.
    if not url.startswith(f"https://github.com/{RELEASES_REPO}/"):
        url = RELEASES_PAGE
    return {"version": str(body.get("tag_name") or ""), "url": url,
            "published_at": body.get("published_at"),
            "highlights": highlights(str(body.get("body") or ""))}


def _result(latest: dict | None, *, checked_at: float | None, error: str | None = None) -> dict:
    version = (latest or {}).get("version") or None
    notes = (latest or {}).get("highlights")
    return {
        "current": APP_VERSION,
        "enabled": is_enabled(),
        "latest": version,
        "newer": bool(version) and is_newer(version),
        "url": (latest or {}).get("url") or RELEASES_PAGE,
        "published_at": (latest or {}).get("published_at"),
        "checked_at": checked_at,
        "error": error,
        "highlights": [str(n) for n in notes][:MAX_HIGHLIGHTS] if isinstance(notes, list) else [],
        # User clicked "Later" in the dialog for this specific version: do not show the dialog again
        # (the small notification bar still shows) until a newer version is available.
        "skipped": bool(version) and _read().get("skipped") == version,
    }


def check(force: bool = False) -> dict:
    with _lock:
        state = _read()
        cached = state.get("latest") if isinstance(state.get("latest"), dict) else None
        checked_at = state.get("checked_at") if isinstance(state.get("checked_at"), (int, float)) else None
        if not force:
            if state.get("enabled") is not True:
                return _result(None, checked_at=None)
            if cached and checked_at and time.time() - checked_at < CHECK_INTERVAL_SECONDS:
                return _result(cached, checked_at=checked_at)
        try:
            latest = fetch_latest()
        except Exception:
            return _result(cached, checked_at=checked_at,
                           error="Could not check for updates (no network connection or GitHub did not respond).")
        now = time.time()
        _write({**state, "latest": latest, "checked_at": now})
        return _result(latest, checked_at=now)


router = APIRouter()


@router.get("/api/app/update")
def get_update(force: bool = False):
    return check(force)


class UpdateCheckSettings(BaseModel):
    enabled: bool


@router.put("/api/app/update-check")
def set_update_check(settings: UpdateCheckSettings):
    with _lock:
        _write({**_read(), "enabled": settings.enabled})
    return {"enabled": is_enabled()}


class UpdateSkip(BaseModel):
    version: str


@router.put("/api/app/update-skip", dependencies=[Depends(require_this_machine)])
def skip_update(body: UpdateSkip):
    with _lock:
        _write({**_read(), "skipped": body.version[:40]})
    return {"skipped": body.version[:40]}
