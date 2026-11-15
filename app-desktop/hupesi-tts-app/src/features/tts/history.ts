import type { HistoryApiEntry } from "./api";
import { notifySaved } from "./desktop-files";
import type { HistoryEntry } from "./types";

// History is stored on disk via the server (apps/history_store.py), NOT in the browser:
// the desktop build runs PyWebView in private_mode, so localStorage/IndexedDB is
// cleared every time the app closes.

const MODES = new Set(["tts", "clone", "conversation", "pdf"]);
const TRAILING_SLASHES = /\/+$/;

export function fromServer(apiUrl: string, e: HistoryApiEntry): HistoryEntry {
  const base = apiUrl.replace(TRAILING_SLASHES, "");
  return {
    id: e.id,
    serverId: e.id,
    audioBlobUrl: `${base}${e.audio_url}`,
    durationSeconds: e.duration_seconds,
    format: e.format,
    mode: MODES.has(e.mode) ? (e.mode as HistoryEntry["mode"]) : "tts",
    subtitlesUrl: e.subtitles_url ? `${base}${e.subtitles_url}` : undefined,
    text: e.text,
    timestamp: e.timestamp,
    voiceId: e.voice_id,
    voiceName: e.voice_name,
  };
}

/**
 * Downloads audio for a history item stored on disk WITHOUT loading it into the browser:
 * the desktop app lets Python copy/convert the MP3 directly from the file, while browsers
 * download it via URL. A ~2-hour recording (~690 MB) passed through a blob/base64 would
 * exceed the browser's memory limits.
 */
export async function downloadHistoryAudio(
  apiUrl: string,
  entryId: string,
  filename: string,
  fmt: "wav" | "mp3"
): Promise<boolean> {
  // biome-ignore lint/suspicious/noExplicitAny: the PyWebView bridge is untyped
  const pywebview = (window as any).pywebview;
  if (typeof pywebview?.api?.save_history_audio === "function") {
    const res = await pywebview.api.save_history_audio(entryId, filename, fmt);
    if (res?.cancelled) {
      return false;
    }
    if (res?.success) {
      notifySaved(res);
      return true;
    }
    throw new Error(res?.error || "Unable to save the file.");
  }
  const base = apiUrl.replace(TRAILING_SLASHES, "");
  const id = encodeURIComponent(entryId);
  const link = document.createElement("a");
  link.href =
    fmt === "mp3"
      ? `${base}/api/history/${id}/mp3`
      : `${base}/api/history/${id}/audio?download=true`;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  return true;
}

/** Returns audio for a history item: uses the available blob or fetches it from the server. */
export async function entryBlob(entry: HistoryEntry): Promise<Blob> {
  if (entry.audioBlob) {
    return entry.audioBlob;
  }
  const response = await fetch(entry.audioBlobUrl);
  if (!response.ok) {
    throw new Error("Unable to download audio for this item.");
  }
  return response.blob();
}

/** Returns the SRT content for a history item, or null if it has no subtitles. */
export async function entrySubtitles(
  entry: HistoryEntry
): Promise<string | null> {
  if (entry.subtitles) {
    return entry.subtitles;
  }
  if (!entry.subtitlesUrl) {
    return null;
  }
  const response = await fetch(entry.subtitlesUrl);
  return response.ok ? response.text() : null;
}

export function hasSubtitles(entry: HistoryEntry): boolean {
  return Boolean(entry.subtitles || entry.subtitlesUrl);
}

/** Unified download file name base for audio and subtitles of the same item. */
export function entryFileBase(entry: HistoryEntry): string {
  return `cit_tts_${entry.voiceName || "voice"}_${entry.timestamp}`;
}

export function srtBlob(srt: string): Blob {
  // UTF-8 BOM: some older players/video editors misread Vietnamese text without it.
  return new Blob([`\uFEFF${srt}`], { type: "application/x-subrip;charset=utf-8" });
}
