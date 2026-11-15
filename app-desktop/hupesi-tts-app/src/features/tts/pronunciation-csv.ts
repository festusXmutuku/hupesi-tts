import type { PronunciationEntry } from "./api";

// Limits must match vieneu_utils/pronunciation.py - exceeding them causes the server to reject the entire save.
export const MAX_WORD_LEN = 64;
export const MAX_READING_LEN = 200;
export const MAX_ENTRIES = 2000;

const HEADER = ["Source word", "Pronunciation", "Case sensitive"];

// Third column: empty = case sensitive (same as the "Add word" default).
const YES = new Set(["c\u00f3", "co", "c", "x", "1", "true", "yes", "y", "enable"]);
const NO = new Set(["kh\u00f4ng", "khong", "k", "0", "false", "no", "n", "disable"]);

/** Sample file: each row illustrates one use case; an empty third column means "yes". */
export const SAMPLE_DICTIONARY_ROWS: string[][] = [
  HEADER,
  ["KTX", "dormitory", "yes"],
  ["SGK", "textbook", "yes"],
  ["AI", "A I", "yes"],
  ["Facebook", "Facebook", "no"],
  ["UN", "United Nations", ""],
];

// BOM written as an escape so it does not appear as an invisible character in source.
const BOM = String.fromCharCode(0xfe_ff);
const NEEDS_QUOTE = /[",\r\n;]/;
const QUOTE = /"/g;
const LINE_BREAK = /\r?\n/;
const SPACES = /\s+/g;

const escapeCell = (value: string) =>
  NEEDS_QUOTE.test(value) ? `"${value.replace(QUOTE, '""')}"` : value;

/** CSV with BOM + CRLF so Excel opens Vietnamese text correctly. */
export function toCsv(rows: string[][]): string {
  return `${BOM}${rows.map((r) => r.map(escapeCell).join(",")).join("\r\n")}\r\n`;
}

export function dictionaryToCsv(entries: PronunciationEntry[]): string {
  return toCsv([
    HEADER,
    ...entries.map((e) => [e.word, e.reading, e.match_case ? "yes" : "no"]),
  ]);
}

/** Vietnamese/European Excel saves CSV with semicolons; detect from the first line. */
function detectDelimiter(text: string): string {
  const firstLine = text.split(LINE_BREAK, 1)[0] ?? "";
  const count = (ch: string) => firstLine.split(ch).length - 1;
  const candidates = [",", ";", "\t"];
  return candidates.reduce((best, ch) => (count(ch) > count(best) ? ch : best));
}

/** Read a quoted field starting after the `"` at `start`; "" represents a literal quote. */
function readQuoted(
  src: string,
  start: number
): { end: number; value: string } {
  let value = "";
  let i = start;
  while (i < src.length) {
    if (src[i] !== '"') {
      value += src[i];
      i++;
    } else if (src[i + 1] === '"') {
      value += '"';
      i += 2;
    } else {
      return { value, end: i + 1 };
    }
  }
  return { value, end: i };
}

/** Parse CSV per RFC 4180: quoted fields may contain delimiters and line breaks. */
export function parseCsv(text: string): string[][] {
  const src = text.startsWith(BOM) ? text.slice(1) : text;
  const delimiter = detectDelimiter(src);
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === '"' && cell === "") {
      const quoted = readQuoted(src, i + 1);
      cell = quoted.value;
      i = quoted.end;
      continue;
    }
    if (ch === delimiter) {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      i += ch === "\r" && src[i + 1] === "\n" ? 1 : 0;
    } else {
      cell += ch;
    }
    i++;
  }
  if (cell !== "" || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

const isHeader = (cells: string[]) => {
  const first = (cells[0] ?? "").trim().toLowerCase();
  return (
    first === "t\u1eeb g\u1ed1c" ||
    first === "from g\u1ed1c" ||
    first === "tu goc" ||
    first === "word" ||
    first === "source word"
  );
};

export interface CsvImportResult {
  entries: PronunciationEntry[];
  /** Line numbers (Excel-style) that were skipped and the reason. */
  skipped: { line: number; reason: string }[];
}

/** Read a dictionary from CSV. If a word appears multiple times, the last row wins. */
export function parseDictionaryCsv(text: string): CsvImportResult {
  const rows = parseCsv(text);
  const byWord = new Map<string, PronunciationEntry>();
  const skipped: CsvImportResult["skipped"] = [];
  rows.forEach((cells, index) => {
    const line = index + 1;
    if (index === 0 && isHeader(cells)) {
      return;
    }
    // Multi-line cells in Excel become a single line: the app import field is single-line.
    const word = (cells[0] ?? "").replace(SPACES, " ").trim();
    const reading = (cells[1] ?? "").replace(SPACES, " ").trim();
    const flag = (cells[2] ?? "").trim().toLowerCase();
    if (!(word || reading)) {
      return; // empty row
    }
    if (!(word && reading)) {
      skipped.push({ line, reason: "missing source word or pronunciation" });
      return;
    }
    if (word.length > MAX_WORD_LEN) {
      skipped.push({ line, reason: `source word exceeds ${MAX_WORD_LEN} characters` });
      return;
    }
    if (reading.length > MAX_READING_LEN) {
      skipped.push({
        line,
        reason: `pronunciation exceeds ${MAX_READING_LEN} characters`,
      });
      return;
    }
    if (flag && !YES.has(flag) && !NO.has(flag)) {
      skipped.push({
        line,
        reason: `the "Case sensitive" column must be "yes" or "no"`,
      });
      return;
    }
    byWord.set(word, { word, reading, match_case: !NO.has(flag) });
  });
  return { entries: [...byWord.values()], skipped };
}
