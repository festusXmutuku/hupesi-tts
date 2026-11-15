import { useEffect, useId, useState } from "react";
import { fetchGpuMemory, type GpuMemoryInfo, saveGpuMemory } from "../api";

/** Re-read after changing models: callers change `key` to remount this component. */
interface GpuMemoryPanelProps {
  apiUrl: string;
  enabled: boolean;
}

const LEVELS_GB = [2, 3, 4, 6, 8, 12, 16, 24];

const gb = (value: number | null | undefined) =>
  value === null || value === undefined
    ? "?"
    : value.toLocaleString("en-US", { maximumFractionDigits: 1 });

/** VRAM limit for GPU models. On 4 GB cards the app could hang on long text
 *  because CUDA graphs were kept; "Auto" is enough for most machines, and the
 *  selector lets users lower it when other heavy apps are running (games, browsers...). */
export function GpuMemoryPanel({ apiUrl, enabled }: GpuMemoryPanelProps) {
  const selectId = useId();
  const [info, setInfo] = useState<GpuMemoryInfo | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!enabled) {
      return;
    }
    const controller = new AbortController();
    setMessage("");
    fetchGpuMemory(apiUrl, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setInfo(result);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setInfo(null);
        }
      });
    return () => controller.abort();
  }, [apiUrl, enabled]);

  if (!info?.available) {
    return null;
  }

  const total = info.total_gb ?? 0;
  const levels = LEVELS_GB.filter((level) => level < total);
  if (info.limit_gb !== null && !levels.includes(info.limit_gb)) {
    levels.push(info.limit_gb);
    levels.sort((a, b) => a - b);
  }

  const handleChange = async (value: string) => {
    const limit = value === "auto" ? null : Number(value);
    setSaving(true);
    setMessage("Applying...");
    try {
      setInfo(await saveGpuMemory(apiUrl, limit));
      setMessage("✓ Applied immediately, no restart required.");
    } catch (error) {
      setMessage(
        error instanceof Error && error.message
          ? error.message
          : "Could not save the VRAM limit. Please try again."
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-card mt-3 space-y-3">
      <label className="settings-card-title" htmlFor={selectId}>
        VRAM limit (GPU)
      </label>
      <p className="settings-help">
        {info.gpu_name} · {gb(total)} GB. Currently using {gb(info.used_gb)} GB, cap{" "}
        {gb(info.effective_limit_gb)} GB. If your machine slows down or freezes on
        long text, choose a lower level - the app will still work, just a bit slower.
      </p>
      <select
        className="w-full rounded-xl border border-border bg-secondary/50 px-3 py-2"
        disabled={saving}
        id={selectId}
        onChange={(event) => handleChange(event.target.value)}
        value={info.limit_gb === null ? "auto" : String(info.limit_gb)}
      >
        <option value="auto">Auto (recommended)</option>
        {levels.map((level) => (
          <option key={level} value={String(level)}>
            Max {level} GB
          </option>
        ))}
      </select>
      {message && (
        <p aria-live="polite" className="settings-help" role="status">
          {message}
        </p>
      )}
    </div>
  );
}
