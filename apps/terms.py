"""Terms of use (DIEU-KHOAN.md): users must accept on first launch.

Acceptance is stored only on the user's machine and is never sent anywhere.
If TERMS_VERSION changes (and the "Terms version" line in DIEU-KHOAN.md), users
will be asked to accept again.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter

TERMS_VERSION = "1.3"

VOICE_RIGHTS_STATEMENT = (
    "I will only use my own voice or a voice for which I have permission from the owner "
    "to clone with AI and use for the agreed purpose. I have the right to use the recording "
    "and will not use the generated voice to impersonate, defraud, mislead, or infringe the "
    "rights or lawful interests of others."
)


def terms_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "terms-accepted.json"


def _terms_file() -> Path:
    # Packaged build: DIEU-KHOAN.md is added via --add-data into _MEIPASS; from source: repo root.
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return root / "DIEU-KHOAN.md"


def terms_text() -> str:
    return _terms_file().read_text(encoding="utf-8")


def accepted() -> bool:
    try:
        data = json.loads(terms_path().read_text(encoding="utf-8"))
        return data.get("terms_version") == TERMS_VERSION
    except (OSError, ValueError, AttributeError):
        return False


def accept() -> None:
    path = terms_path()
    path.parent.mkdir(parents=True, exist_OK=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump({"terms_version": TERMS_VERSION}, f)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_OK=True)


router = APIRouter()


@router.get("/api/terms")
def get_terms():
    return {"version": TERMS_VERSION, "accepted": accepted(), "text": terms_text()}


@router.post("/api/terms/accept")
def post_accept():
    accept()
    return {"version": TERMS_VERSION, "accepted": True}
