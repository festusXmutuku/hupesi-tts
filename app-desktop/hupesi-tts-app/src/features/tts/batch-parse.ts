// Split "Batch generation" text into items by markers [P1], [P2]...
//
// A marker must appear at the start of a line; its content may be on the same line
// ("[P1] Hello") or on the lines below, continuing until the next marker. Only the
// exact [P<number>] form is recognized so emotion tags like [laugh] or pause tags
// like [pause 2s] in the content are not affected.

export interface BatchItem {
  /** Display label and file name, e.g. "P1"; "Preamble" for content before the first marker. */
  label: string;
  text: string;
}

export interface BatchParseResult {
  /** So moc khong co noi dung (bi bo qua). */
  emptyMarkers: number;
  /** false = khong tim thay moc nao: ca van ban la MOT muc. */
  hasMarkers: boolean;
  items: BatchItem[];
}

const MARKER = /^[ \t]*\[\s*[Pp]\s*(\d{1,4})\s*\][ \t]*:?[ \t]*/;
const NEWLINES = /\r\n?/g;
export const PREAMBLE_LABEL = "Preamble";

export function parseBatchScript(input: string): BatchParseResult {
  const items: BatchItem[] = [];
  let emptyMarkers = 0;
  let hasMarkers = false;
  let label: string | null = null;
  let buffer: string[] = [];

  const flush = () => {
    const text = buffer.join("\n").trim();
    if (text) {
      items.push({ label: label ?? PREAMBLE_LABEL, text });
    } else if (label !== null) {
      emptyMarkers++;
    }
    buffer = [];
  };

  for (const line of input.replace(NEWLINES, "\n").split("\n")) {
    const m = MARKER.exec(line);
    if (m) {
      flush();
      hasMarkers = true;
      label = `P${Number(m[1])}`;
      buffer.push(line.slice(m[0].length));
    } else {
      buffer.push(line);
    }
  }
  flush();
  if (!hasMarkers && items.length === 1) {
    items[0].label = "P1";
  }
  return { emptyMarkers, hasMarkers, items };
}

/** File name for item `index` (0-based): "001-P1.wav" - the sequence number preserves order when opening the zip. */
export function batchFileName(
  index: number,
  total: number,
  label: string,
  ext: string
): string {
  const width = Math.max(3, String(total).length);
  const safe = label.replace(/[\\/:*?"<>|\s]+/g, "-");
  return `${String(index + 1).padStart(width, "0")}-${safe}.${ext}`;
}
