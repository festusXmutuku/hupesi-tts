"""
Extract text from Word .docx files using only the standard library (zipfile + ElementTree).

A .docx file is a zip archive containing XML; the main content lives in ``word/document.xml``.
We do not use python-docx to avoid adding a dependency (and avoid having to declare it for
PyInstaller - if missing, the .exe would fail at import on a user's machine).

Word paragraph boundaries are preserved: one paragraph per line, with paragraphs separated by a
blank line. The TTS pipeline splits paragraphs on line breaks, which is an advantage over PDF
(PDF line breaks follow page layout and can create unnatural pauses mid-sentence).

Extracted: paragraphs, text in hyperlinks/tracked inserts/content controls, and tables (one row
per line, cells joined with ", "). Skipped: deleted text (tracked delete), text boxes,
headers/footers, and notes - parts that are usually not meant to be read aloud.
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_EP = "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}"

# Guard against zip bombs: document.xml may be decompressed to at most this many bytes.
MAX_XML_BYTES = 64 * 1024 * 1024

# Branches not meant to be read aloud: text boxes/drawings nested in runs, deleted text.
_SKIP = {f"{_W}txbxContent", f"{_W}del", f"{_W}delText", f"{_W}instrText",
         f"{_W}footnoteReference", f"{_W}endnoteReference"}


class DocxError(ValueError):
    """The file is not a valid .docx - user-facing error shown directly to the user."""


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_XML_BYTES:
        raise DocxError("The document is too large to extract.")
    return zf.read(info)


def _run_text(node: ET.Element, out: list[str]) -> None:
    """Collect a paragraph's text in document order, skipping branches in _SKIP."""
    for child in node:
        tag = child.tag
        if tag in _SKIP:
            continue
        if tag == f"{_W}t":
            out.append(child.text or "")
        elif tag == f"{_W}tab":
            out.append(" ")
        elif tag in (f"{_W}br", f"{_W}cr"):
            out.append("\n")
        elif tag == f"{_W}noBreakHyphen":
            out.append("-")
        else:
            _run_text(child, out)


def _paragraph(p: ET.Element) -> str:
    parts: list[str] = []
    _run_text(p, parts)
    lines = ("".join(parts)).split("\n")
    return "\n".join(" ".join(line.split()) for line in lines).strip()


def _table(tbl: ET.Element) -> list[str]:
    rows: list[str] = []
    # Only DIRECT child rows/cells (do not use iter): a table nested inside a cell is already
    # collected by _blocks(tc), so iter would collect it twice.
    for tr in tbl.findall(f"{_W}tr"):
        cells = []
        for tc in tr.findall(f"{_W}tc"):
            text = " ".join(" ".join(_blocks(tc)).split())
            if text:
                cells.append(text)
        if cells:
            rows.append(", ".join(cells))
    return rows


def _blocks(container: ET.Element) -> list[str]:
    """Text blocks (paragraphs / table rows) of a body, table cell, or content control."""
    out: list[str] = []
    for child in container:
        if child.tag == f"{_W}p":
            text = _paragraph(child)
            if text:
                out.append(text)
        elif child.tag == f"{_W}tbl":
            out.extend(_table(child))
        elif child.tag == f"{_W}sdt":
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                out.extend(_blocks(content))
    return out


def _page_count(zf: zipfile.ZipFile) -> int:
    """Page count recorded by Word at save time (docProps/app.xml); 0 if unavailable."""
    try:
        raw = _read_member(zf, "docProps/app.xml")
        pages = ET.fromstring(raw).find(f"{_EP}Pages") if raw else None
        return int(pages.text) if pages is not None and pages.text else 0
    except (ET.ParseError, ValueError, DocxError):
        return 0


def extract_docx(data: bytes) -> tuple[str, int]:
    """Return ``(text, page_count)``. Corrupt or non-docx file -> ``DocxError``."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise DocxError(
            "The file is not a valid .docx format (it may be corrupt or have a renamed extension)."
        ) from e
    with zf:
        raw = _read_member(zf, "word/document.xml")
        if raw is None:
            raise DocxError("The .docx file has no text content (missing word/document.xml).")
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            raise DocxError(f"The .docx file content is corrupt: {e}") from e
        body = root.find(f"{_W}body")
        blocks = _blocks(body) if body is not None else []
        return "\n\n".join(blocks), _page_count(zf)
