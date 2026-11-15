"""
User-defined pronunciation dictionary (substitution at the TEXT level).

The sea-g2p normalizer treats ALL-CAPS abbreviations as English and spells them out letter by
letter in an English style ("AI" -> ``<en>a i</en>``). Writing the desired reading as plain
text (for example "ây ai") means the normalizer leaves it as-is and the model speaks it correctly
in a Vietnamese voice, so we only need to replace word -> reading BEFORE text enters the pipeline.
We do not use sea-g2p's ``phoneme_dict`` because it requires IPA input, which most users cannot
write themselves.

Matching rules:
- Match WHOLE WORDS only: "AI" does not affect "OpenAI" or "AIDS".
- Case-sensitive by default: "AI" does not affect the pronoun "ai". Entries with
  ``match_case=False`` match any capitalization.
- Longer words are tried before shorter ones ("API" wins over "AI" regardless of input order).
- Emotion tags like ``[cười]`` / ``<|emotion_k|>`` are never touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

# Emotion tags - must be preserved verbatim so phonemize_text can recognize them (see
# _EMOTION_SPLIT_RE in phonemize_text.py).
_PROTECTED_RE = re.compile(r"(\[[^\]]+\]|<\|emotion_\d+\|>)")

MAX_WORD_LEN = 64
MAX_READING_LEN = 200
MAX_ENTRIES = 2000

# Default set when the user has not saved a dictionary yet - also shown as an example in the UI.
# Only includes abbreviations the normalizer does not know and would spell out letter by letter
# (e.g. "KTX"). Common abbreviations (CLB, THPT, HLV...) are already read correctly by the
# normalizer and do not need to be added.
DEFAULT_ENTRIES: list[dict] = [
    {"word": "KTX", "reading": "ký túc xá", "match_case": True},
    {"word": "SGK", "reading": "sách giáo khoa", "match_case": True},
]


@dataclass(frozen=True)
class PronunciationEntry:
    word: str
    reading: str
    match_case: bool = True


def validate_entries(raw: Iterable[Mapping]) -> list[PronunciationEntry]:
    """Validate and normalize a list of dictionary entries. Raises ``ValueError`` on invalid input."""
    entries: list[PronunciationEntry] = []
    exact: set[str] = set()        # case-sensitive words, kept as-is
    insensitive: set[str] = set()  # case-insensitive words, lowercased
    all_folded: set[str] = set()   # all words, lowercased
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Entry {i} is invalid.")
        word = str(item.get("word") or "").strip()
        reading = " ".join(str(item.get("reading") or "").split())
        match_case = item.get("match_case", True)
        if not isinstance(match_case, bool):
            raise ValueError(f"Entry {i}: 'match_case' must be true/false.")
        if not word:
            raise ValueError(f"Entry {i}: the word to replace must not be empty.")
        if not reading:
            raise ValueError(f"Entry {i} ('{word}'): the reading must not be empty.")
        if len(word) > MAX_WORD_LEN:
            raise ValueError(f"Entry {i}: the word is longer than {MAX_WORD_LEN} characters.")
        if len(reading) > MAX_READING_LEN:
            raise ValueError(f"Entry {i} ('{word}'): the reading is longer than {MAX_READING_LEN} characters.")
        if _PROTECTED_RE.search(word):
            raise ValueError(f"Entry {i}: cannot replace emotion tag '{word}'.")
        # Two entries conflict when they can match the same string: compare lowercased if either
        # one is case-insensitive.
        folded = word.lower()
        clash = (
            (word in exact or folded in insensitive)
            if match_case
            else folded in all_folded
        )
        if clash:
            raise ValueError(f"The word '{word}' is declared more than once.")
        (exact if match_case else insensitive).add(word if match_case else folded)
        all_folded.add(folded)
        entries.append(PronunciationEntry(word, reading, match_case))
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f"The dictionary allows at most {MAX_ENTRIES} entries.")
    return entries


class PronunciationDict:
    """Precompiled replacer: a single regex for all dictionary entries."""

    def __init__(self, entries: Iterable[PronunciationEntry]):
        self.entries = list(entries)
        self._exact = {e.word: e.reading for e in self.entries if e.match_case}
        self._folded = {e.word.lower(): e.reading for e in self.entries if not e.match_case}
        if not self.entries:
            self._re = None
            return
        # Longer before shorter: regex alternation takes the FIRST matching branch.
        ordered = sorted(self.entries, key=lambda e: len(e.word), reverse=True)
        alts = [
            re.escape(e.word) if e.match_case else f"(?i:{re.escape(e.word)})"
            for e in ordered
        ]
        # "Whole word" boundaries based on Unicode \w (including Vietnamese diacritics), not \b,
        # because \b does not work when a word starts/ends with characters like "." ("A.I.").
        self._re = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)")

    def _reading_for(self, m: re.Match) -> str:
        s = m.group(0)
        if s in self._exact:
            return self._exact[s]
        return self._folded.get(s.lower(), s)

    def apply(self, text: str) -> str:
        if not text or self._re is None:
            return text
        parts = _PROTECTED_RE.split(text)
        for i in range(0, len(parts), 2):  # odd indices are emotion tags - skip them
            parts[i] = self._re.sub(self._reading_for, parts[i])
        return "".join(parts)


def apply_pronunciations(text: str, entries: Iterable[Mapping]) -> str:
    """One-shot helper: validate ``entries`` then apply replacements to ``text``."""
    return PronunciationDict(validate_entries(entries)).apply(text)
