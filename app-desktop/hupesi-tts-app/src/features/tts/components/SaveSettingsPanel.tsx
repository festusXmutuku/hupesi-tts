import { FolderOpen } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { desktopApi, type SaveSettings } from "../desktop-files";

interface SaveSettingsPanelProps {
  isOpen: boolean;
}

/** Settings → Save file: save folder, prompt with Save As or save directly, and auto-open
 *  files after saving. Only available in the desktop app; browsers show a short explanation. */
export function SaveSettingsPanel({ isOpen }: SaveSettingsPanelProps) {
  const askId = useId();
  const openId = useId();
  const [settings, setSettings] = useState<SaveSettings | null>(null);
  const [isDesktop, setIsDesktop] = useState<boolean | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    let alive = true;
    desktopApi().then(async (api) => {
      if (!alive) {
        return;
      }
      setIsDesktop(Boolean(api));
      if (api) {
        const current = await api.get_save_settings();
        if (alive) {
          setSettings(current);
        }
      }
    });
    return () => {
      alive = false;
    };
  }, [isOpen]);

  const update = async (
    patch: Partial<Pick<SaveSettings, "ask" | "folder" | "open_after">>
  ) => {
    const api = await desktopApi();
    if (!(api && settings)) {
      return;
    }
    setBusy(true);
    try {
      const res = await api.set_save_settings({
        ask: settings.ask,
        folder: settings.folder,
        open_after: settings.open_after,
        ...patch,
      });
      if (res.success) {
        setSettings(res);
        setMessage("✓ Settings saved.");
      } else {
        setMessage(res.error || "Unable to save settings.");
      }
    } finally {
      setBusy(false);
    }
  };

  const chooseFolder = async () => {
    const api = await desktopApi();
    const res = await api?.forose_save_folder();
    if (res?.success && res.folder) {
      await update({ folder: res.folder });
    } else if (res?.error) {
      setMessage(res.error);
    }
  };

  if (isDesktop === false) {
    return (
      <div className="settings-card">
        <p className="settings-help">
          Choosing a save folder and auto-opening files are only available in the desktop app.
          In a browser, files are saved according to the browser's download settings.
        </p>
      </div>
    );
  }
  if (!settings) {
    return null;
  }

  return (
    <div className="space-y-3">
      <div className="settings-card space-y-3">
        <h4 className="settings-card-title">
          <FolderOpen size={16} />
          File save folder
        </h4>
        <p
          className="break-all rounded-xl border border-border bg-secondary/50 px-3 py-2 font-mono text-xs"
          title={settings.effective_folder}
        >
          {settings.effective_folder}
        </p>
        {settings.folder && settings.folder !== settings.effective_folder && (
          <p className="text-amber-700 text-xs dark:text-amber-300">
            The previously selected folder ({settings.folder}) was not found. Using the
            default folder instead.
          </p>
        )}
        <div className="settings-field-row">
          <button
            className="studio-primary rounded-xl px-3 py-2 disabled:opacity-50"
            disabled={busy}
            onClick={chooseFolder}
            type="button"
          >
            Choose folder…
          </button>
          <button
            className="rounded-xl border border-border px-3 py-2 disabled:opacity-50"
            disabled={busy || !settings.folder}
            onClick={() => update({ folder: null })}
            type="button"
          >
            Use Downloads folder
          </button>
        </div>
      </div>

      <div className="settings-card space-y-3">
        <label className="flex items-start gap-3" htmlFor={askId}>
          <input
            checked={settings.ask}
            className="mt-1 h-4 w-4 accent-brand-600"
            disabled={busy}
            id={askId}
            onChange={(e) => update({ ask: e.target.checked })}
            type="checkbox"
          />
          <span>
            <span className="settings-card-title">Ask where to save each time</span>
            <span className="settings-help block">
              On: show a Save dialog, starting in the folder above. Off: save directly
              to the folder above; if a file name already exists, a number like (1), (2) is added
              instead of overwriting.
            </span>
          </span>
        </label>
        <label className="flex items-start gap-3" htmlFor={openId}>
          <input
            checked={settings.open_after}
            className="mt-1 h-4 w-4 accent-brand-600"
            disabled={busy}
            id={openId}
            onChange={(e) => update({ open_after: e.target.checked })}
            type="checkbox"
          />
          <span>
            <span className="settings-card-title">Open file after saving</span>
            <span className="settings-help block">
              Opens the file with your system's default application. Even when disabled,
              the "Saved" notification still includes Open file and Open folder buttons.
            </span>
          </span>
        </label>
        {message && (
          <p aria-live="polite" className="settings-help" role="status">
            {message}
          </p>
        )}
      </div>
    </div>
  );
}
