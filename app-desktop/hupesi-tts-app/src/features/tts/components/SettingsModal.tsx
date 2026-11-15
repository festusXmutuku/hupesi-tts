import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  Cpu,
  ExternalLink,
  FolderOpen,
  Layers,
  Mic,
  Plug,
  RefreshCw,
  Server,
  Settings2,
  Wifi,
  X,
} from "lucide-react";
import type React from "react";
import { useEffect, useId, useRef, useState } from "react";
import type { AppUpdateInfo } from "../api";
import { checkServerHealth, fetchServerPort, saveServerPort } from "../api";
import type { AppSettings, VoiceOption } from "../types";
import { UpdateCheckToggle } from "./AppUpdate";
import { DefaultVoicePicker } from "./DefaultVoicePicker";
import { GpuMemoryPanel } from "./GpuMemoryPanel";
import { ModelManager } from "./ModelManager";
import { RemoteAccessPanel } from "./RemoteAccessPanel";
import { RestartNotice, useAppRestart } from "./RestartNotice";
import { SaveSettingsPanel } from "./SaveSettingsPanel";

const TRAILING_SLASH = /\/$/;

/** When the UI is served by the server itself (desktop app, browser opening port 8001), it
 *  always calls that same server - the address field is only needed for Vite dev or when
 *  already pointing to another server. */
function needsServiceUrl(current: string): boolean {
  const { origin, protocol } = window.location;
  return (
    import.meta.env.DEV ||
    !protocol.startsWith("http") ||
    current.replace(TRAILING_SLASH, "") !== origin
  );
}

// Only list public APIs, matching /openapi.json and hupesi-tts.md.
// When adding a public API path, update this list.
const API_ENDPOINTS = [
  { method: "GET", path: "/health", desc: "Server status and model name" },
  { method: "GET", path: "/voices", desc: "List available voices" },
  { method: "POST", path: "/stream", desc: "Real-time streaming (PCM)" },
  { method: "POST", path: "/api/tts/generate", desc: "Convert text to WAV/MP3" },
  {
    method: "POST",
    path: "/api/tts/generate-with-subtitles",
    desc: "Audio with .srt subtitles",
  },
  {
    method: "POST",
    path: "/api/tts/conversation",
    desc: "Multi-speaker conversation",
  },
  {
    method: "GET",
    path: "/hupesi-tts.md",
    desc: "Markdown documentation for AI agents",
  },
] as const;

const SETTINGS_SECTIONS = [
  {
    id: "models",
    label: "AI Models",
    icon: Layers,
    description: "Choose a model that fits your machine and reading needs.",
  },
  {
    id: "voice",
    label: "Voice",
    icon: Mic,
    description: "Voice selected by default when opening the app.",
  },
  {
    id: "files",
    label: "File saving",
    icon: FolderOpen,
    description: "Save folder for audio, subtitles, and opening files after saving.",
  },
  {
    id: "connection",
    label: "Connection",
    icon: Server,
    description: "Application server port.",
  },
  {
    id: "network",
    label: "LAN network",
    icon: Wifi,
    description: "Manage API access permissions from other machines on the same network.",
  },
  {
    id: "updates",
    label: "Updates & info",
    icon: RefreshCw,
    description: "Options for checking for new versions and audio information.",
  },
  {
    id: "api",
    label: "API integration",
    icon: Plug,
    description: "Connect Hupesi TTS with your applications and workflows.",
  },
] as const;
type SettingsSection = (typeof SETTINGS_SECTIONS)[number]["id"];
const FOCUSABLE =
  'button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]';

function handleDialogKey(
  event: KeyboardEvent,
  dialog: HTMLElement,
  onClose: () => void
) {
  if (event.key === "Escape") {
    event.stopPropagation();
    onClose();
    return;
  }
  if (event.key !== "Tab") {
    return;
  }
  const elements = Array.from(
    dialog.querySelectorAll<HTMLElement>(FOCUSABLE)
  ).filter((element) => element.getClientRects().length > 0);
  const first = elements[0];
  const last = elements.pop();
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last?.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first?.focus();
  }
}

interface SettingsModalProps {
  initialSection?: "models";
  isOpen: boolean;
  onClose: () => void;
  onModelSwitched?: (modelName: string) => void;
  onSaveSettings: (settings: AppSettings) => void;
  /** Result of a version check after enabling automatic update checks. */
  onUpdateChecked?: (info: AppUpdateInfo) => void;
  settings: AppSettings;
  voices: VoiceOption[];
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  initialSection,
  onClose,
  settings,
  onSaveSettings,
  onModelSwitched,
  onUpdateChecked,
  voices,
}) => {
  const titleId = useId();
  const urlId = useId();
  const portId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [section, setSection] = useState<SettingsSection>("models");
  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setSection(initialSection ?? "models");
    const previous = document.activeElement;
    closeRef.current?.focus();
    return () => {
      if (previous instanceof HTMLElement) {
        previous.focus();
      }
    };
  }, [isOpen, initialSection]);
  useEffect(() => {
    if (section && isOpen) {
      contentRef.current?.scrollTo({ top: 0 });
    }
  }, [section, isOpen]);
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!(isOpen && dialog)) {
      return;
    }
    const handleKey = (event: KeyboardEvent) =>
      handleDialogKey(event, dialog, onClose);
    dialog.addEventListener("keydown", handleKey);
    return () => dialog.removeEventListener("keydown", handleKey);
  }, [isOpen, onClose]);
  const [apiUrl, setApiUrl] = useState(settings.apiUrl);
  useEffect(() => {
    if (isOpen) {
      setApiUrl(settings.apiUrl);
      setTestStatus("idle");
      setTestMessage("");
    }
  }, [isOpen, settings.apiUrl]);
  const [testStatus, setTestStatus] = useState<
    "idle" | "testing" | "success" | "error"
  >("idle");
  const [testMessage, setTestMessage] = useState("");
  const [copiedExample, setCopiedExample] = useState(false);
  const [serverPort, setServerPort] = useState("8001");
  const [activePort, setActivePort] = useState<number | null>(null);
  const [portMessage, setPortMessage] = useState("");
  const [portReady, setPortReady] = useState(false);
  const [portSaving, setPortSaving] = useState(false);
  // If a different port is saved than the one currently running, a restart notice is needed.
  const [pendingPort, setPendingPort] = useState<number | null>(null);
  const appRestart = useAppRestart(settings.apiUrl);
  // Incremented when the model changes: GpuMemoryPanel remounts to read the new GPU status.
  const [modelVersion, setModelVersion] = useState(0);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const controller = new AbortController();
    setPortReady(false);
    setPortMessage("Reading server port configuration...");
    fetchServerPort(settings.apiUrl, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) {
          return;
        }
        setServerPort(String(result.port));
        setActivePort(result.active_port);
        setPortReady(true);
        setPendingPort(result.port === result.active_port ? null : result.port);
        setPortMessage("");
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setPortMessage(
            "Unable to read port configuration. Please connect to the server and reopen Settings."
          );
        }
      });
    return () => controller.abort();
  }, [isOpen, settings.apiUrl]);

  const handleSavePort = async (andRestart = false) => {
    const port = Number(serverPort);
    if (
      !(/^\d+$/.test(serverPort) && Number.isInteger(port)) ||
      port < 1 ||
      port > 65_535
    ) {
      setPortMessage("Enter a port number from 1 to 65535.");
      return;
    }
    setPortSaving(true);
    setPortMessage("Saving server port...");
    try {
      const result = await saveServerPort(settings.apiUrl, port);
      setActivePort(result.active_port);
      if (result.port === result.active_port) {
        setPendingPort(null);
        setPortMessage(
          `✓ Saved. Port ${result.port} is already running; no restart needed.`
        );
        return;
      }
      setPendingPort(result.port);
      setPortMessage("");
      if (andRestart) {
        await appRestart.restart();
      }
    } catch {
      setPortMessage(
        "Unable to save the server port. Check the connection and try again."
      );
    } finally {
      setPortSaving(false);
    }
  };

  // Python example that works on Windows; avoids Unix-specific line continuation syntax.
  const apiExample = `import json, urllib.request

body = json.dumps({
    "text": "Hello from another application",
    "voice_id": "Sample Voice",
    "speed": 1.0,
}).encode()

req = urllib.request.Request(
    "${apiUrl.replace(/\/$/, "")}/api/tts/generate",
    data=body,
    headers={"Content-Type": "application/json"},
    # Calling from another machine: replace 127.0.0.1 with this machine's IP and add
    # "X-API-Key": "<key in Settings>" to headers.
)
with urllib.request.urlopen(req) as r:
    open("speech.wav", "wb").write(r.read())`;

  const handleCopyExample = () => {
    navigator.clipboard.writeText(apiExample);
    setCopiedExample(true);
    setTimeout(() => setCopiedExample(false), 2000);
  };

  if (!isOpen) {
    return null;
  }

  const handleTestConnection = async () => {
    setTestStatus("testing");
    setTestMessage("Checking connection...");
    try {
      const res = await checkServerHealth(apiUrl.trim());
      if (res.status === "loading") {
        setTestStatus("success");
        setTestMessage(
          `Connected to the service. ${res.message || "Model is loading, please wait."}`
        );
        return;
      }
      if (res.status === "error") {
        setTestStatus("error");
        setTestMessage(
          `Connected to the service, but the model encountered an error: ${res.message || "Could not load model."}`
        );
        return;
      }
      if (res.status !== "OK") {
        throw new Error("Server returned an invalid status.");
      }
      setTestStatus("success");
      setTestMessage(
        `Connection successful! Server: ${res.engine} (Sample rate: ${res.sample_rate / 1000} kHz)`
      );
    } catch (e) {
      setTestStatus("error");
      setTestMessage(
        `Connection error: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  // Each section saves in place; the service address has its own Save button like the port.
  const handleSaveUrl = () => {
    // After saving, the button disables itself (the address already matches the config), which is enough feedback.
    onSaveSettings({ ...settings, apiUrl: apiUrl.trim() });
  };

  const currentSection =
    SETTINGS_SECTIONS.find((item) => item.id === section) ??
    SETTINGS_SECTIONS[0];

  const apiBase = apiUrl.replace(TRAILING_SLASH, "");
  const showServiceUrl = needsServiceUrl(settings.apiUrl);

  return (
    <div
      aria-labelledby={titleId}
      aria-modal="true"
      className="settings-overlay fixed inset-0 z-[60]"
      ref={dialogRef}
      role="dialog"
    >
      <div className="settings-dialog">
        <header className="settings-header">
          <div className="flex items-center gap-3">
            <span className="settings-header-icon">
              <Settings2 size={22} />
            </span>
            <div>
              <h2 id={titleId}>Settings</h2>
              <p>Hupesi TTS · Customize your workspace</p>
            </div>
          </div>
          <button
            aria-label="Close settings"
            className="studio-icon-button"
            onClick={onClose}
            ref={closeRef}
            type="button"
          >
            <X size={20} />
          </button>
        </header>
        <div className="settings-body">
          <nav aria-label="Settings categories" className="settings-nav">
            <span className="settings-nav-label">Customize app</span>
            {SETTINGS_SECTIONS.map((item) => (
              <button
                aria-current={section === item.id ? "page" : undefined}
                key={item.id}
                onClick={() => setSection(item.id)}
                type="button"
              >
                <item.icon aria-hidden="true" size={18} />
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="settings-content" ref={contentRef}>
            <div className="settings-section-title">
              <h3>{currentSection.label}</h3>
              <p>{currentSection.description}</p>
            </div>
            <div className="settings-section" hidden={section !== "models"}>
              <ModelManager
                apiUrl={apiUrl}
                embedded
                enabled={isOpen}
                initialExpanded
                onModelSwitched={(modelName) => {
                  setModelVersion((v) => v + 1);
                  onModelSwitched?.(modelName);
                }}
              />
              <GpuMemoryPanel
                apiUrl={apiUrl}
                enabled={isOpen}
                key={modelVersion}
              />
            </div>
            <div className="settings-section" hidden={section !== "voice"}>
              <div className="settings-card">
                <DefaultVoicePicker
                  active={isOpen && section === "voice"}
                  onChange={(defaultVoiceId) =>
                    onSaveSettings({ ...settings, defaultVoiceId })
                  }
                  value={settings.defaultVoiceId ?? ""}
                  voices={voices}
                />
              </div>
            </div>
            <div className="settings-section" hidden={section !== "connection"}>
              {/* Service address: the desktop app always connects to its own server (and clears
                  localStorage on each open), so only show this when the UI runs elsewhere. */}
              {showServiceUrl && (
                <div className="settings-card space-y-3">
                  <label className="settings-card-title" htmlFor={urlId}>
                    Voice service address
                  </label>
                  <p className="settings-help">
                    Keep the default address if the service runs on this machine.
                  </p>
                  <div className="settings-field-row">
                    <input
                      className="min-w-0 flex-1 rounded-xl border border-border bg-secondary/50 px-3 py-2 font-mono text-xs outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20"
                      id={urlId}
                      onChange={(e) => {
                        setApiUrl(e.target.value);
                        setTestStatus("idle");
                      }}
                      placeholder="http://127.0.0.1:8001"
                      type="text"
                      value={apiUrl}
                    />
                    <button
                      className="flex items-center gap-1 rounded-xl border border-border bg-secondary px-3 py-2 font-medium transition-all hover:bg-secondary/80"
                      disabled={testStatus === "testing"}
                      onClick={handleTestConnection}
                      type="button"
                    >
                      {testStatus === "testing" ? (
                        <>
                          <RefreshCw className="h-3.5 w-3.5 animate-spin text-amber-500" />
                          <span>Checking...</span>
                        </>
                      ) : (
                        <span>Check</span>
                      )}
                    </button>
                    <button
                      className="studio-primary rounded-xl px-3 py-2 disabled:opacity-50"
                      disabled={apiUrl.trim() === settings.apiUrl}
                      onClick={handleSaveUrl}
                      type="button"
                    >
                      Save
                    </button>
                  </div>
                  {testStatus !== "idle" && (
                    <div
                      aria-live="polite"
                      className={`mt-2 flex items-start gap-2 rounded-xl border p-2.5 text-[12px] ${
                        testStatus === "success"
                          ? "border-brand-500/30 bg-brand-500/10 text-brand-600 dark:text-brand-400"
                          : testStatus === "error"
                            ? "border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-400"
                            : "bg-muted text-muted-foreground"
                      }`}
                      role="status"
                    >
                      {testStatus === "success" ? (
                        <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      ) : (
                        <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      )}
                      <span>{testMessage}</span>
                    </div>
                  )}
                </div>
              )}

              <div className="settings-card space-y-3">
                <label className="settings-card-title" htmlFor={portId}>
                  Port on app startup
                </label>
                <p className="settings-help">
                  Default 8001
                  {activePort === null ? "" : ` · Running: ${activePort}`}.
                  If the port is busy, the app selects the next available port.
                </p>
                <div className="settings-field-row">
                  <input
                    className="min-w-0 flex-1 rounded-xl border border-border bg-secondary/50 px-3 py-2"
                    disabled={!portReady || portSaving}
                    id={portId}
                    max={65_535}
                    min={1}
                    onChange={(event) => setServerPort(event.target.value)}
                    step={1}
                    type="number"
                    value={serverPort}
                  />
                  <button
                    className="rounded-xl border border-border px-3 py-2 disabled:opacity-50"
                    disabled={!portReady || portSaving}
                    onClick={() => setServerPort("8001")}
                    type="button"
                  >
                    Default 8001
                  </button>
                  <button
                    className="studio-primary rounded-xl bg-brand-600 px-3 py-2 text-white disabled:opacity-50"
                    disabled={!portReady || portSaving}
                    onClick={() => handleSavePort()}
                    type="button"
                  >
                    {portSaving ? "Saving..." : "Save port"}
                  </button>
                </div>
                {appRestart.supported && (
                  <button
                    className="w-full rounded-xl border border-brand-600/60 px-3 py-2 font-semibold text-brand-700 hover:bg-brand-500/10 disabled:opacity-50 dark:text-brand-300"
                    disabled={!portReady || portSaving || appRestart.restarting}
                    onClick={() => handleSavePort(true)}
                    type="button"
                  >
                    Save &amp; restart
                  </button>
                )}
                {appRestart.restarting ? (
                  <p
                    aria-live="polite"
                    className="font-medium text-amber-700 dark:text-amber-300"
                    role="status"
                  >
                    Restarting… the window will close and reopen automatically in a few seconds.
                  </p>
                ) : pendingPort === null ? null : (
                  <RestartNotice
                    apiUrl={settings.apiUrl}
                    message={`Saved port ${pendingPort}.`}
                  />
                )}
                {appRestart.error && (
                  <p className="text-rose-600 dark:text-rose-400" role="alert">
                    {appRestart.error}
                  </p>
                )}
                {portMessage && (
                  <p
                    aria-live="polite"
                    className="text-muted-foreground"
                    role="status"
                  >
                    {portMessage}
                  </p>
                )}
              </div>
            </div>
            <div className="settings-section" hidden={section !== "network"}>
              <RemoteAccessPanel apiUrl={settings.apiUrl} isOpen={isOpen} />
            </div>
            <div className="settings-section" hidden={section !== "files"}>
              <SaveSettingsPanel isOpen={isOpen} />
            </div>
            <div className="settings-section" hidden={section !== "updates"}>
              <UpdateCheckToggle
                apiUrl={settings.apiUrl}
                isOpen={isOpen}
                onChecked={onUpdateChecked}
              />

              {/* Audio feature information. */}
              <div className="settings-card space-y-3">
                <h4 className="settings-card-title">
                  <Cpu size={16} />
                  Audio info
                </h4>
                <dl className="settings-facts">
                  <div>
                    <dt>Sample rate</dt>
                    <dd>{settings.sampleRate.toLocaleString("en-US")} Hz</dd>
                  </div>
                  <div>
                    <dt>Voice Cloning</dt>
                    <dd>Read the consent sentence via microphone</dd>
                  </div>
                  <div>
                    <dt>Conversation</dt>
                    <dd>Multiple speakers</dd>
                  </div>
                  <div>
                    <dt>Export formats</dt>
                    <dd>WAV, MP3</dd>
                  </div>
                </dl>
              </div>
            </div>
            <div className="settings-section" hidden={section !== "api"}>
              {/* Documentation comes first in this section: integrators need it first. */}
              <div className="settings-card settings-docs">
                <div className="settings-docs-text">
                  <h4 className="settings-card-title">
                    <BookOpen size={16} />
                    API documentation
                  </h4>
                  <p className="settings-help">
                    Full parameter reference and the ability to try calls directly in your browser.
                  </p>
                </div>
                <div className="settings-docs-actions">
                  <a
                    className="settings-docs-primary"
                    href={`${apiBase}/docs`}
                    rel="noreferrer"
                    target="_blank"
                  >
                    Open documentation (Swagger)
                    <ExternalLink size={14} />
                  </a>
                  <a
                    className="settings-docs-secondary"
                    href={`${apiBase}/hupesi-tts.md`}
                    rel="noreferrer"
                    target="_blank"
                    title="Markdown documentation for AI agents (ChatGPT, Claude…)"
                  >
                    AI agent version (.md)
                    <ExternalLink size={14} />
                  </a>
                </div>
              </div>
              <div className="settings-card">
                <div className="space-y-4">
                  <p className="settings-help">
                    While the app is running, it exposes a local API server. Other software{" "}
                    <span className="text-foreground">
                      on this same machine
                    </span>{" "}
                    can call it to generate speech without an API key. For other machines
                    on the LAN: enable{" "}
                    <span className="text-foreground">
                      Allow other machines to access the API
                    </span>{" "}
                    in the LAN network section, replace{" "}
                    <code className="font-mono">127.0.0.1</code> with this machine's IP address,
                    and include the{" "}
                    <code className="font-mono">X-API-Key</code> header.
                  </p>

                  <div className="settings-api-table">
                    <table className="w-full">
                      <thead>
                        <tr>
                          <th scope="col">Method</th>
                          <th scope="col">Path</th>
                          <th scope="col">Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {API_ENDPOINTS.map((e) => (
                          <tr key={e.path + e.method}>
                            <td>
                              <span
                                className="settings-method"
                                data-method={e.method}
                              >
                                {e.method}
                              </span>
                            </td>
                            <td className="font-mono">{e.path}</td>
                            <td>{e.desc}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <h4 className="settings-card-title">
                        Speech generation example
                      </h4>
                      <button
                        className="flex items-center gap-1 rounded-md border border-border/60 bg-secondary px-2 py-1 font-medium transition-colors hover:bg-secondary/80"
                        onClick={handleCopyExample}
                        type="button"
                      >
                        {copiedExample ? (
                          <>
                            <CheckCircle2 className="h-3 w-3 text-brand-500" />
                            <span className="text-brand-500">Copied</span>
                          </>
                        ) : (
                          <span>Copy</span>
                        )}
                      </button>
                    </div>
                    <pre className="select-all overflow-x-auto rounded-lg bg-zinc-950 p-3 font-mono text-[12.5px] text-zinc-200 leading-relaxed">
                      {apiExample}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* All changes are saved in place; the dialog footer only needs a Close button. */}
        <div className="settings-footer">
          <button
            className="ml-auto rounded-xl border border-border px-4 py-1.5 text-muted-foreground transition-all hover:bg-secondary hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
