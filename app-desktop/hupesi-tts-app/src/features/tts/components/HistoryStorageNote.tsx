import { AlertTriangle, FolderOpen, HardDrive } from "lucide-react";
import type React from "react";
import { useEffect, useState } from "react";
import { fetchHistoryInfo, type HistoryInfo, openHistoryFolder } from "../api";

// History is NOT deleted automatically (previously, exceeding 300 items / 1 GB would
// silently delete the oldest items). This note lets users know how much space history
// uses, where it is stored, and reminds them to clean up when it's too large or the disk is low.

const formatBytes = (bytes: number) => {
  if (bytes >= 1024 ** 3) {
    return `${(bytes / 1024 ** 3).toLocaleString("en-US", { maximumFractionDigits: 1 })} GB`;
  }
  return `${Math.max(0, Math.round(bytes / 1024 ** 2)).toLocaleString("en-US")} MB`;
};

export const HistoryStorageNote: React.FC<{
  apiUrl: string;
  /** Change this value (e.g. the number of history items) to reload storage usage. */
  refreshKey: number;
}> = ({ apiUrl, refreshKey }) => {
  const [info, setInfo] = useState<HistoryInfo | null>(null);
  const [error, setError] = useState("");

  // biome-ignore lint/correctness/useExhaustiveDependencies: refreshKey is only used to reload when history changes
  useEffect(() => {
    const controller = new AbortController();
    fetchHistoryInfo(apiUrl, controller.signal)
      .then(setInfo)
      .catch(() => undefined); // old backend / disconnected: hide the note
    return () => controller.abort();
  }, [apiUrl, refreshKey]);

  if (!info) {
    return null;
  }

  const handleOpen = () => {
    setError("");
    openHistoryFolder(apiUrl).catch((e) =>
      setError(e instanceof Error ? e.message : String(e))
    );
  };

  let reason = "";
  if (info.disk_free !== null && info.disk_free < 2 * 1024 ** 3) {
    reason = "Disk space is running low. ";
  } else if (info.warn) {
    reason = "Generated recordings are taking up a lot of space. ";
  }

  return (
    <div
      className={`space-y-1 rounded-xl border px-3 py-2 text-xs ${
        info.warn
          ? "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300"
          : "border-border/70 bg-card/60 text-muted-foreground"
      }`}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        {info.warn ? (
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <HardDrive className="h-3.5 w-3.5 shrink-0" />
        )}
        <span>
          <span className="text-foreground">{info.count} items</span> ·{" "}
          {formatBytes(info.bytes)}
          {info.disk_free !== null &&
            ` · ${formatBytes(info.disk_free)} free on disk`}
        </span>
        <button
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-semibold text-sky-600 transition-colors hover:bg-sky-500/10 dark:text-sky-400"
          onClick={handleOpen}
          title={info.folder}
          type="button"
        >
          <FolderOpen className="h-3.5 w-3.5" />
          Open folder
        </button>
      </div>
      <p className="leading-relaxed">
        {reason}
        History is not deleted automatically. Download the recordings you want to keep and delete old ones to free up disk space.
      </p>
      {error && (
        <p className="text-rose-600 dark:text-rose-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};
