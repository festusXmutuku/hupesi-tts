// Clean text extracted from PDF / Word before reading.
//
// Principle: only remove things that are CLEARLY layout noise, never touch content.
// Math symbols ("1 + 1", "a - b", "- 5 degrees", "x * y") must be preserved, so:
// - bullet markers are removed only when at the start of a line and followed by a letter;
// - never change punctuation/symbols in the middle of a line;
// - "..." (ellipsis/pause) is kept; only sequences of 4+ dots are removed.

export interface CleanupStats {
  blankLines: number;
  bullets: number;
  joinedLines: number;
  placeholders: number;
}

export interface CleanupResult {
  stats: CleanupStats;
  text: string;
}

// Fill-in placeholders in forms: "......", "……", "____".
const PLACEHOLDER = /\.{4,}|(?:…\s*){2,}|_{3,}/g;
// Common dash bullets: only when followed by a space and a letter.
const DASH_BULLET = /^[-*+–\u2014]\s+(?=\p{L})/u;
// Symbols used only as bullets (not math operators): no space required.
const SYMBOL_BULLET = /^[•▪◦●○■□➢►✓✔]\s*(?=\p{L})/u;
const SPACES = /[ \t ]+/g;
// A line ending with sentence punctuation/closing brackets is a complete thought; don't join with the next line.
const LINE_END = /[.!?:;…"”')\]]$/;
const STARTS_LOWER = /^\p{Ll}/u;
const NEWLINES = /\r\n?/g;

interface Line {
  hadBullet: boolean;
  text: string;
}

export function cleanExtractedText(input: string): CleanupResult {
  const stats: CleanupStats = {
    blankLines: 0,
    bullets: 0,
    joinedLines: 0,
    placeholders: 0,
  };

  const lines: Line[] = [];
  for (const raw of input.replace(NEWLINES, "\n").split("\n")) {
    let text = raw.replace(PLACEHOLDER, () => {
      stats.placeholders++;
      return " ";
    });
    text = text.replace(SPACES, " ").trim();

    let hadBullet = false;
    const withoutBullet = text
      .replace(DASH_BULLET, "")
      .replace(SYMBOL_BULLET, "");
    if (withoutBullet !== text) {
      hadBullet = true;
      stats.bullets++;
      text = withoutBullet;
    }

    if (!text) {
      stats.blankLines++;
      continue;
    }
    lines.push({ hadBullet, text });
  }

  // Join lines broken mid-sentence (PDFs break at page width). Do not join into list
  // items: "- room a" / "- room b" are separate points and joining would lose a pause.
  const out: string[] = [];
  for (const line of lines) {
    // biome-ignore lint/style/useAtIndex: tsconfig lib does not include Array.prototype.at yet
    const prev = out.length ? out[out.length - 1] : undefined;
    if (
      prev !== undefined &&
      !line.hadBullet &&
      !LINE_END.test(prev) &&
      STARTS_LOWER.test(line.text)
    ) {
      out[out.length - 1] = `${prev} ${line.text}`;
      stats.joinedLines++;
    } else {
      out.push(line.text);
    }
  }

  return { stats, text: out.join("\n") };
}

/** User-facing result summary; "" if nothing changed. */
export function describeCleanup(stats: CleanupStats): string {
  const parts = [
    stats.blankLines && `${stats.blankLines} blank line${stats.blankLines === 1 ? "" : "s"}`,
    stats.bullets && `${stats.bullets} bullet${stats.bullets === 1 ? "" : "s"}`,
    stats.placeholders && `${stats.placeholders} fill-in placeholder${stats.placeholders === 1 ? "" : "s"}`,
  ].filter(Boolean);
  const joined = stats.joinedLines
    ? `joined ${stats.joinedLines} line${stats.joinedLines === 1 ? "" : "s"} broken mid-sentence`
    : "";
  if (parts.length === 0 && !joined) {
    return "";
  }
  const removed = parts.length ? `removed ${parts.join(", ")}` : "";
  return `Cleaned up: ${[removed, joined].filter(Boolean).join("; ")}.`;
}
