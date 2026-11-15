"""Generate build/DIEU-KHOAN.txt (UTF-8 with BOM) from DIEU-KHOAN.md for the Inno Setup
installer's terms page - Inno displays .txt files verbatim, so markdown markers are removed.

DIEU-KHOAN.md is manually hard-wrapped (~88 characters). Inno's terms panel is narrower and
wraps on its own, so keeping those breaks makes each sentence break awkwardly. Therefore each
paragraph / list item is unwrapped into a single line so the panel can wrap to its own width.

Usage: python make_terms_txt.py   (called automatically by make_installer.bat)
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (script lives in scripts/)
_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")


def _unwrap(text: str) -> str:
    """Join manually wrapped lines within the same paragraph / list item; keep blank lines,
    headings, and the start of each list item."""
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        starts_block = (
            not stripped
            or stripped.startswith("#")
            or _ITEM.match(line)
            or not out
            or not out[-1].strip()
            or out[-1].lstrip().startswith("#")
        )
        if starts_block:
            out.append(line.rstrip())
        else:
            out[-1] = f"{out[-1].rstrip()} {stripped}"
    return "\n".join(out)


def write(out: Path = ROOT / "build" / "DIEU-KHOAN.txt") -> Path:
    text = (ROOT / "DIEU-KHOAN.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    text = _unwrap(text)
    # Uppercase section headings (##) so they stand out in plain text; keep the document title (#).
    text = re.sub(r"^##+\s*(.+)$", lambda m: m.group(1).upper(), text, flags=re.M)
    text = re.sub(r"^#\s*", "", text, flags=re.M)
    text = re.sub(r"^- ", "• ", text, flags=re.M)
    text = text.replace("**", "").replace("`", "")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text.replace("\n", "\r\n"), encoding="utf-8-sig", newline="")
    return out


if __name__ == "__main__":
    print(f"[+] {write()}")
