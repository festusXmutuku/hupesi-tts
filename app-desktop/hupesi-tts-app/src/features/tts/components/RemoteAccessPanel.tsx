import {
  AlertTriangle,
  Copy,
  Eye,
  EyeOff,
  KeyRound,
  RefreshCw,
  Wifi,
} from "lucide-react";
import type React from "react";
import { useEffect, useState } from "react";
import {
  fetchRemoteAccess,
  type RemoteAccessInfo,
  regenerateRemoteKey,
  saveRemoteAccess,
} from "../api";
import { RestartNotice } from "./RestartNotice";

interface RemoteAccessPanelProps {
  apiUrl: string;
  isOpen: boolean;
}

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function statusOf(info: RemoteAccessInfo): {
  text: string;
  tone: "OK" | "muted";
  restart?: string;
} {
  if (info.enabled && info.listening_all) {
    return {
      text: "Enabled — other devices can access when remote access is on.",
      tone: "OK",
    };
  }
  if (info.enabled) {
    return {
      restart: "Saved: remote access enabled.",
      text: "",
      tone: "muted",
    };
  }
  if (info.listening_all) {
    return {
      restart: "Saved: remote access disabled.",
      text: "",
      tone: "muted",
    };
  }
  return {
    text: "Disabled — only apps on this machine can access the server.",
    tone: "muted",
  };
}

export const RemoteAccessPanel: React.FC<RemoteAccessPanelProps> = ({
  apiUrl,
  isOpen,
}) => {
  const [info, setInfo] = useState<RemoteAccessInfo | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const controller = new AbortController();
    setError("");
    fetchRemoteAccess(apiUrl, controller.signal)
      .then(setInfo)
      .catch((e) => {
        if (!controller.signal.aborted) {
          setError(errorText(e));
        }
      });
    return () => controller.abort();
  }, [apiUrl, isOpen]);

  const run = async (action: () => Promise<RemoteAccessInfo>) => {
    setBusy(true);
    setError("");
    try {
      setInfo(await action());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const copy = (label: string, value: string) => {
    navigator.clipboard.writeText(value).then(() => {
      setCopied(label);
      setTimeout(() => setCopied(null), 1500);
    });
  };

  const handleRegenerate = () => {
    if (
      // biome-ignore lint/suspicious/noAlert: simple confirmation before irreversible action
      !window.confirm(
        "Generate a new key? Devices using the old key will be rejected until updated."
      )
    ) {
      return;
    }
    run(() => regenerateRemoteKey(apiUrl));
  };

  const status = info ? statusOf(info) : null;
  const mainIp = info?.primary_address ?? info?.addresses[0] ?? null;
  const otherIps = info?.addresses.filter((ip) => ip !== mainIp) ?? [];
  const firstUrl = mainIp
    ? `http://${mainIp}:${info?.port}`
    : `http://<this-machine-ip>:${info?.port ?? 8001}`;

  return (
    <div className="settings-card space-y-3">
      <label className="settings-toggle">
        <span aria-hidden="true" className="settings-toggle-icon">
          <Wifi size={18} />
        </span>
        <span className="settings-toggle-text">
          <span>Allow access from other devices (LAN)</span>
          <span>
            Lets other devices open the UI and use the server on your network.
            Use only on trusted networks.
          </span>
        </span>
        <input
          checked={Boolean(info?.enabled)}
          className="studio-switch"
          disabled={!info || busy}
          onChange={(e) =>
            run(() => saveRemoteAccess(apiUrl, e.target.checked))
          }
          type="checkbox"
        />
      </label>

      {status?.restart && (
        <>
          <RestartNotice apiUrl={apiUrl} message={status.restart} />
          {info?.enabled && (
            <p className="text-[12px] text-muted-foreground leading-relaxed">
              After restart, Windows may ask for firewall permission the first
              time: forose <span>Allow</span> on <span>Private</span>{" "}
              networks.
            </p>
          )}
        </>
      )}
      {status && !status.restart && (
        <p
          aria-live="polite"
          className="settings-status"
          data-tone={status.tone}
          role="status"
        >
          <span aria-hidden="true" className="settings-status-dot" />
          {status.text}
        </p>
      )}

      {info && (
        <div className="space-y-2 text-[12px]">
          <div className="space-y-1">
            <span className="flex items-center gap-1 font-semibold text-foreground">
              <KeyRound className="h-3.5 w-3.5 text-amber-500" /> API key
            </span>
            <div className="flex gap-1.5">
              <input
                className="min-w-0 flex-1 rounded-lg border border-border bg-secondary/50 px-2.5 py-1.5 font-mono text-[12px]"
                readOnly
                type={showKey ? "text" : "password"}
                value={info.api_key}
              />
              <button
                aria-label={showKey ? "Hide key" : "Show key"}
                className="rounded-lg border border-border px-2 hover:bg-secondary"
                onClick={() => setShowKey(!showKey)}
                title={showKey ? "Hide key" : "Show key"}
                type="button"
              >
                {showKey ? (
                  <EyeOff className="h-3.5 w-3.5" />
                ) : (
                  <Eye className="h-3.5 w-3.5" />
                )}
              </button>
              <button
                className="flex items-center gap-1 rounded-lg border border-border px-2 hover:bg-secondary"
                onClick={() => copy("key", info.api_key)}
                type="button"
              >
                <Copy className="h-3.5 w-3.5" />
                {copied === "key" ? "Copied" : "Copy"}
              </button>
              <button
                className="flex items-center gap-1 rounded-lg border border-border px-2 hover:bg-secondary disabled:opacity-50"
                disabled={busy}
                onClick={handleRegenerate}
                title="Generate a new key and invalidate the old one"
                type="button"
              >
                <RefreshCw className="h-3.5 w-3.5" /> New key
              </button>
            </div>
          </div>

          <div className="space-y-1">
            <span className="font-semibold text-foreground">
              Address for other devices
            </span>
            {mainIp ? (
              <p className="font-mono font-semibold text-foreground">
                http://{mainIp}:{info.port}
              </p>
            ) : (
              <p className="text-muted-foreground">
                No LAN IP found — is this machine connected to a network?
              </p>
            )}
            {otherIps.length > 0 && (
              <p className="text-muted-foreground leading-relaxed">
                Other IPs on this machine:{" "}
                <span className="font-mono">{otherIps.join(", ")}</span> —
                these are often virtual adapters (VirtualBox, Hyper-V, VPN) and
                may not be reachable from other devices.
              </p>
            )}
          </div>

          <div className="space-y-1">
            <span className="font-semibold text-foreground">
              Test from another machine (cmd or PowerShell)
            </span>
            <div className="flex gap-1.5">
              <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded-lg bg-zinc-950 px-2.5 py-1.5 font-mono text-[12px] text-zinc-200">
                curl.exe -H "X-API-Key: {showKey ? info.api_key : "<key>"}"{" "}
                {firstUrl}/health
              </code>
              <button
                className="flex items-center gap-1 rounded-lg border border-border px-2 hover:bg-secondary"
                onClick={() =>
                  copy(
                    "curl",
                    `curl.exe -H "X-API-Key: ${info.api_key}" ${firstUrl}/health`
                  )
                }
                type="button"
              >
                <Copy className="h-3.5 w-3.5" />
                {copied === "curl" ? "Copied" : "Copy"}
              </button>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              Send the key via <code className="font-mono">X-API-Key</code>{" "}
              (or <code className="font-mono">Authorization: Bearer</code>).
            </p>
          </div>

          <p className="flex items-start gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/5 px-2.5 py-2 text-amber-700 leading-relaxed dark:text-amber-300">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              Enable only on trusted networks. Traffic is not encrypted over
              plain HTTP — to expose it over the Internet, use an HTTPS tunnel
              (Cloudflare Tunnel, ngrOK) instead of opening a router port.
            </span>
          </p>
        </div>
      )}

      {error && (
        <p className="text-rose-600 dark:text-rose-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};
