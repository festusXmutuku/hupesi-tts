import { AudioLines, CheckCircle2, Download, RefreshCw, X } from "lucide-react";
import type React from "react";
import { useEffect, useId, useRef, useState } from "react";
import {
  type AppUpdateInfo,
  checkAppUpdate,
  saveUpdateCheck,
  skipAppUpdate,
} from "../api";

// Update check: only notifies and links to the download page, never auto-downloads/installs
// (installer ~365 MB, unsigned so SmartScreen may prompt; a failed install could break the app).

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** "New version available" banner below the tab bar. Shown only when a newer version exists. */
export const UpdateBanner: React.FC<{
  info: AppUpdateInfo | null;
  onDismiss: () => void;
}> = ({ info, onDismiss }) => {
  if (!info?.newer) {
    return null;
  }
  return (
    <div className="slide-in-from-top-2 mx-6 mt-3 flex animate-in items-center justify-between gap-3 rounded-xl border border-brand-500/30 bg-brand-500/10 p-3 text-brand-700 text-xs shadow-sm dark:text-brand-300">
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <Download className="h-4 w-4 shrink-0" />
        <span>
          Version <strong>{info.latest}</strong> is available (you are using{" "}
          {info.current}).
        </span>
        <a
          className="font-bold underline underline-offset-2 hover:text-brand-500"
          href={info.url}
          rel="noreferrer"
          target="_blank"
        >
          View changes &amp; download
        </a>
      </div>
      <button
        className="shrink-0 rounded-lg p-1 transition-colors hover:bg-brand-500/20"
        onClick={onDismiss}
        title="Later"
        type="button"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
};

/** "New version available" dialog, shown once per version. "Later" remembers the
 *  skipped version on the server so it won't reopen next time (the small banner remains);
 *  the X button only closes it for this session. Opens the download page only; never auto-downloads/installs. */
export const UpdateDialog: React.FC<{
  apiUrl: string;
  info: AppUpdateInfo;
  onClose: () => void;
}> = ({ apiUrl, info, onClose }) => {
  const titleId = useId();
  const primaryRef = useRef<HTMLAnchorElement>(null);

  useEffect(() => {
    primaryRef.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const handleLater = () => {
    if (info.latest) {
      skipAppUpdate(apiUrl, info.latest).catch(() => undefined);
    }
    onClose();
  };

  return (
    <div
      aria-labelledby={titleId}
      aria-modal="true"
      className="update-dialog-overlay fixed inset-0 z-[70]"
      role="dialog"
    >
      <div className="update-dialog fade-in-50 zoom-in-95 animate-in">
        <div className="flex items-start gap-3">
          <span className="update-dialog-logo">
            <AudioLines size={26} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="font-bold text-base text-foreground" id={titleId}>
                Hupesi TTS
              </h2>
              <span className="update-dialog-badge">Update</span>
            </div>
            <p className="mt-0.5 font-mono text-muted-foreground text-xs">
              {info.current} → {info.latest}
            </p>
          </div>
          <button
            aria-label="Close"
            className="rounded-lg p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="mt-4 text-foreground/80 text-sm">
          Version <strong>{info.latest}</strong> is available with new features and bug fixes.
          Installing over the old version keeps your cloned voices, generated audio, and settings.
        </p>
        {info.highlights && info.highlights.length > 0 && (
          <ul className="update-dialog-list">
            {info.highlights.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        )}

        <div className="update-dialog-footer">
          <a
            className="font-medium text-foreground text-sm underline underline-offset-4 hover:text-brand-600"
            href={info.url}
            rel="noreferrer"
            target="_blank"
          >
            What&apos;s new
          </a>
          <div className="flex gap-2">
            <button
              className="rounded-lg border border-border px-4 py-2 font-semibold text-foreground text-xs transition-colors hover:bg-secondary"
              onClick={handleLater}
              type="button"
            >
              Later
            </button>
            <a
              className="studio-primary inline-flex items-center gap-1.5 rounded-lg px-4 py-2 font-semibold text-xs"
              href={info.url}
              onClick={onClose}
              ref={primaryRef}
              rel="noreferrer"
              target="_blank"
            >
              <Download className="h-3.5 w-3.5" />
              Download new version
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

/** "Check for updates" button in the About tab (queries GitHub immediately, even if auto-check is disabled). */
export const UpdateCheckButton: React.FC<{
  apiUrl: string;
  info: AppUpdateInfo | null;
  onChecked: (info: AppUpdateInfo) => void;
  variant?: "default" | "sidebar";
}> = ({ apiUrl, info, onChecked, variant = "default" }) => {
  const [busy, setBusy] = useState(false);
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState("");

  const handleCheck = async () => {
    setBusy(true);
    setError("");
    try {
      const result = await checkAppUpdate(apiUrl, true);
      onChecked(result);
      setChecked(true);
      if (result.error) {
        setError(result.error);
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className={
        variant === "sidebar"
          ? "studio-update-check"
          : "flex flex-wrap items-center gap-2 text-xs"
      }
    >
      <button
        className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-secondary/60 px-2.5 py-1 font-semibold text-foreground transition-colors hover:bg-secondary disabled:opacity-50"
        disabled={busy}
        onClick={handleCheck}
        type="button"
      >
        <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} />
        {busy ? "Checking..." : "Check for updates"}
      </button>
      {error && (
        <span className="text-rose-600 dark:text-rose-400" role="alert">
          {error}
        </span>
      )}
      {info?.newer && (
        <a
          className="font-bold text-brand-600 underline underline-offset-2 dark:text-brand-400"
          href={info.url}
          rel="noreferrer"
          target="_blank"
        >
          Download version {info.latest}
        </a>
      )}
      {!(error || info?.error) &&
        (checked || info?.checked_at) &&
        !info?.newer && (
          <span className="inline-flex items-center gap-1 text-brand-600 dark:text-brand-400">
            <CheckCircle2 className="h-3.5 w-3.5" />
            You are using the latest version.
          </span>
        )}
    </div>
  );
};

/** Version from backend, shared by the About tab and Settings. */
export const SidebarFooter: React.FC<{
  apiUrl: string;
  info: AppUpdateInfo | null;
  onChecked: (info: AppUpdateInfo) => void;
}> = ({ apiUrl, info, onChecked }) => (
  <footer className="studio-sidebar-footer">
    <div className="studio-sidebar-credit">
      <span>
        by{" "}
        <a
          className="rounded-sm font-bold underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-current focus-visible:outline-offset-2"
          href="https://github.com/festusxmutukugithub"
          rel="noopener noreferrer"
          target="_blank"
          title="Festus Mutuku personal page (opens in a new tab)"
        >
          Festus Mutuku
        </a>
      </span>
      <span className="studio-sidebar-version" title="Current version">
        {info ? `v${info.current}` : "Connecting…"}
      </span>
    </div>
    <UpdateCheckButton
      apiUrl={apiUrl}
      info={info}
      onChecked={onChecked}
      variant="sidebar"
    />
  </footer>
);

/** Settings toggle: automatically check for updates when the app opens (the only time the app contacts the Internet). */
export const UpdateCheckToggle: React.FC<{
  apiUrl: string;
  isOpen: boolean;
  /** When just enabled: check immediately so the new-version banner appears now, without waiting for the next launch. */
  onChecked?: (info: AppUpdateInfo) => void;
}> = ({ apiUrl, isOpen, onChecked }) => {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const controller = new AbortController();
    // Not forced: use the saved result, do not query GitHub again.
    checkAppUpdate(apiUrl, false, controller.signal)
      .then((info) => setEnabled(info.enabled))
      .catch(() => {
        if (!controller.signal.aborted) {
          setEnabled(null);
        }
      });
    return () => controller.abort();
  }, [apiUrl, isOpen]);

  const handleChange = async (value: boolean) => {
    setError("");
    try {
      const saved = await saveUpdateCheck(apiUrl, value);
      setEnabled(saved);
      if (saved) {
        checkAppUpdate(apiUrl)
          .then((info) => onChecked?.(info))
          .catch(() => undefined);
      }
    } catch (e) {
      setError(errorText(e));
    }
  };

  return (
    <div className="settings-card space-y-2">
      <label className="settings-toggle">
        <span aria-hidden="true" className="settings-toggle-icon">
          <RefreshCw size={18} />
        </span>
        <span className="settings-toggle-text">
          <strong>Automatically check for updates on startup</strong>
          <span>
            Checks GitHub once per day and does not send text, audio, or
            machine information. When off, the app stays fully offline; you can
            still check manually in the About tab.
          </span>
        </span>
        <input
          checked={enabled === true}
          className="studio-switch"
          disabled={enabled === null}
          onChange={(e) => handleChange(e.target.checked)}
          type="checkbox"
        />
      </label>
      {error && (
        <p className="text-rose-600 dark:text-rose-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};
