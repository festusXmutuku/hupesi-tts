import {
  AlertTriangle,
  Ban,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Download,
  Feather,
  Languages,
  Layers,
  RefreshCw,
  Zap,
} from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { useModelDownload } from "../hooks/useModelDownload";
import type { LanguagePack, ModelOption } from "../types";
import { ModelOperationStatus } from "./model-operation-status";

interface ModelManagerProps {
  apiUrl: string;
  embedded?: boolean;
  enabled: boolean;
  initialExpanded?: boolean;
  onModelSwitched?: (modelName: string) => void;
}

function docGon(n: number): string {
  if (!n) {
    return "";
  }
  return n < 1024 ** 3
    ? `${Math.round(n / 1024 ** 2)} MB`
    : `${(n / 1024 ** 3).toFixed(1)} GB`;
}

interface TrangThai {
  kind: "ready" | "missing" | "setup" | "blocked";
  text: string;
}

function trangThai(m: ModelOption): TrangThai {
  if (m.compatible === false) {
    return {
      kind: "blocked",
      text: m.unavailable_reason ?? "This machine cannot run this model",
    };
  }
  if (m.downloaded) {
    if (m.needs_gpu_setup) {
      return { kind: "setup", text: "Model downloaded · GPU support not installed" };
    }
    if (m.needs_addon_setup) {
      return { kind: "setup", text: "Model downloaded · Audio separation add-on not installed" };
    }
    return { kind: "ready", text: "Downloaded" };
  }
  return { kind: "missing", text: `Not downloaded · ${docGon(m.approx_bytes ?? 0)}` };
}

// Highlight labels in model names; other parts in parentheses are regular specs.
const NHAN_NOI_BAT = new Map([
  ["Recommended", "brand"],
  ["Khuy\u1ebfn ngh\u1ecb", "brand"],
  ["Fastest", "brand"],
  ["Nhanh nh\u1ea5t", "brand"],
  ["Experimental", "warn"],
  ["Th\u1eed nghi\u1ec7m", "warn"],
]);

const TEN_CO_NGOAC = /^(.*?)\s*\((.*)\)\s*$/;
const NGAN_NHAN = /\s+[·-]\s+/;

/** "VieNeu-TTS v3 Turbo (INT8 CPU - Recommended)" -> title + tags inside parentheses. */
function tachTen(name: string): { tags: string[]; title: string } {
  const match = name.match(TEN_CO_NGOAC);
  if (!match) {
    return { title: name, tags: [] };
  }
  return {
    title: match[1],
    tags: match[2].split(NGAN_NHAN).filter(Boolean),
  };
}

function bieuTuong(m: ModelOption) {
  if (m.device === "cuda" || m.device === "gpu" || m.name.includes("GPU")) {
    return Zap;
  }
  if (m.name.startsWith("Supertonic") || m.name.startsWith("Kokoro")) {
    return Languages;
  }
  if (m.name.includes("Nano")) {
    return Feather;
  }
  return Cpu;
}

/** Missing libraries (GPU or Kokoro's audio separation add-on): clicking will start the download path;
 * the server skips files already in cache and installs only what's missing. */
function canCai(m: ModelOption): boolean {
  return Boolean(m.needs_gpu_setup || m.needs_addon_setup);
}

interface ModelRowProps {
  dangDung: boolean;
  disabled: boolean;
  model: ModelOption;
  onAction: (m: ModelOption) => void;
  onAddPack: (m: ModelOption, pack: LanguagePack) => void;
  pending?: "downloading" | "applying";
}

/** Additional language packs (Kokoro: Japanese, Chinese) - only shown when the model is ready. */
function PackList({
  model,
  disabled,
  onAddPack,
}: Pick<ModelRowProps, "model" | "disabled" | "onAddPack">) {
  if (!(model.packs?.length && model.downloaded) || model.needs_addon_setup) {
    return null;
  }
  return (
    <div className="model-card-packs">
      <span>Add language:</span>
      {model.packs.map((p) =>
        p.installed ? (
          <span className="model-tag" data-highlight="brand" key={p.code}>
            {p.name} · Installed
          </span>
        ) : (
          <button
            className="model-pack-button"
            disabled={disabled}
            key={p.code}
            onClick={() => onAddPack(model, p)}
            type="button"
          >
            <Download size={12} />
            <span>
              {p.name} (~{docGon(p.approx_bytes)})
            </span>
          </button>
        )
      )}
    </div>
  );
}

function nhanNut(
  model: ModelOption,
  pending?: "downloading" | "applying"
): string {
  if (pending === "applying") {
    return "Applying…";
  }
  if (pending) {
    return "Downloading…";
  }
  if (model.needs_gpu_setup) {
    return "Install GPU support";
  }
  if (!model.downloaded) {
    return "Download & Apply";
  }
  return model.needs_addon_setup ? "Install audio separation" : "Apply";
}

function ModelRow({
  model,
  dangDung,
  disabled,
  onAction,
  onAddPack,
  pending,
}: ModelRowProps) {
  const khongChayDuoc = model.compatible === false;
  const { title, tags } = tachTen(model.name);
  const tt = trangThai(model);
  const Icon = bieuTuong(model);
  // Short note, shown in full next to the status and wrapping as needed.
  const ghiChu = [
    model.needs_gpu_setup && !khongChayDuoc
      ? "Installs additional GPU libraries; requires network access and at least 12 GB of free space."
      : "",
    model.needs_addon_setup && !khongChayDuoc
      ? "Installs the audio separation add-on (about 22 MB, downloaded from PyPI)."
      : "",
    model.note ?? "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <div
      className="model-card"
      data-active={dangDung || undefined}
      data-blocked={khongChayDuoc || undefined}
    >
      <span aria-hidden="true" className="model-card-icon">
        <Icon size={20} />
      </span>
      <div className="model-card-body">
        <div className="model-card-head">
          <p className="model-card-title">{title}</p>
          {tags.map((tag) => (
            <span
              className="model-tag"
              data-highlight={NHAN_NOI_BAT.get(tag)}
              key={tag}
            >
              {tag}
            </span>
          ))}
          {model.supports_cloning && !khongChayDuoc && (
            <span className="model-tag">Voice Cloning</span>
          )}
        </div>
        <p className="model-card-status" data-kind={tt.kind}>
          {tt.kind === "ready" && <CheckCircle2 size={14} />}
          {tt.kind === "missing" && <Download size={14} />}
          {tt.kind === "setup" && <AlertTriangle size={14} />}
          {tt.kind === "blocked" && <Ban size={14} />}
          <span className="model-card-status-text">{tt.text}</span>
          {ghiChu && <span className="model-card-note">{ghiChu}</span>}
        </p>
        <PackList disabled={disabled} model={model} onAddPack={onAddPack} />
      </div>

      {dangDung && (
        <span className="model-card-current">
          <Check size={14} strokeWidth={3} />
          In use
        </span>
      )}
      {!(dangDung || khongChayDuoc) && (
        <button
          className="model-card-action"
          data-download={!model.downloaded || canCai(model) || undefined}
          disabled={disabled}
          onClick={() => onAction(model)}
          type="button"
        >
          {pending ? (
            <RefreshCw className="motion-safe:animate-spin" size={14} />
          ) : (
            !model.downloaded && <Download size={14} />
          )}
          <span>{nhanNut(model, pending)}</span>
        </button>
      )}
    </div>
  );
}

/**
 * Model list with three distinct states: downloaded, not downloaded, and not
 * compatible with this machine. Native `<select>` elements can't include badges and
 * per-row buttons, so a row list is used instead.
 */
export const ModelManager: React.FC<ModelManagerProps> = ({
  embedded = false,
  initialExpanded = false,
  apiUrl,
  enabled,
  onModelSwitched,
}) => {
  // Collapsed by default, like the API section - most users don't need to open it every
  // time they visit Settings, and useModelDownload only starts polling when actually expanded.
  const [isExpanded, setIsExpanded] = useState(initialExpanded);
  const {
    models,
    current,
    status,
    busy,
    cancelling,
    error,
    download,
    apply,
    cancel,
  } = useModelDownload(apiUrl, enabled && isExpanded);
  const previousModelRef = useRef<string | null>(null);
  const operationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (busy) {
      operationRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [busy]);

  // Installing a language pack reloads Kokoro under the same name, so the effect below
  // doesn't detect a model change: notify separately so new voices and languages appear.
  // Only notify when this open session observes the transition from installing to ready. The
  // server retains the "ready" state from previous installs, and Settings recreates ModelManager
  // on each open, so relying on the current state would trigger a reload every time.
  const previousStateRef = useRef(status.state);
  useEffect(() => {
    const was = previousStateRef.current;
    previousStateRef.current = status.state;
    const vuaCaiXong =
      (was === "downloading" || was === "applying") &&
      status.state === "ready" &&
      Boolean(status.pack);
    if (vuaCaiXong && current) {
      onModelSwitched?.(current);
    }
  }, [status.state, status.pack, current, onModelSwitched]);

  // Only update the voice list after the server has actually applied the model.
  useEffect(() => {
    if (!current || previousModelRef.current === current) {
      return;
    }
    const previous = previousModelRef.current;
    previousModelRef.current = current;
    if (previous !== null) {
      onModelSwitched?.(current);
    }
  }, [current, onModelSwitched]);

  const dangChay =
    status.state === "downloading" || status.state === "applying";

  const xuLy = (m: ModelOption) => {
    void (m.downloaded && !canCai(m) ? apply(m.id) : download(m.id));
  };
  const operationModel = models.find((m) => m.id === status.model_id);
  // Retry the exact operation that failed: if a language pack was being downloaded, download that pack again.
  const thuLai = (): void => {
    if (!operationModel) {
      return;
    }
    if (status.pack) {
      download(operationModel.id, status.pack);
    } else {
      xuLy(operationModel);
    }
  };
  const coTheThuLai =
    operationModel !== undefined && operationModel.compatible !== false;

  return (
    <div
      className={
        embedded
          ? "settings-model-manager"
          : "rounded-xl border border-border p-3"
      }
    >
      <button
        className="flex w-full items-center justify-between gap-2 text-left"
        hidden={embedded}
        onClick={() => setIsExpanded(!isExpanded)}
        type="button"
      >
        <span className="flex items-center gap-1.5 font-semibold text-foreground">
          <Layers className="h-4 w-4 text-brand-500" />
          <span>Manage AI models</span>
        </span>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-3 space-y-4">
          <div className="model-operation-anchor" ref={operationRef}>
            <ModelOperationStatus
              busy={busy}
              cancelling={cancelling}
              error={error}
              modelName={
                operationModel?.name ?? status.model_id ?? "Models AI"
              }
              onCancel={cancel}
              onRetry={coTheThuLai ? thuLai : undefined}
              status={status}
            />
          </div>
          <div className="model-list">
            {models.map((m) => (
              <ModelRow
                dangDung={current === m.id}
                disabled={dangChay || busy}
                key={m.id}
                model={m}
                onAction={xuLy}
                onAddPack={(model, p) => {
                  download(model.id, p.code);
                }}
                pending={
                  status.model_id === m.id && dangChay
                    ? (status.state as "downloading" | "applying")
                    : undefined
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
