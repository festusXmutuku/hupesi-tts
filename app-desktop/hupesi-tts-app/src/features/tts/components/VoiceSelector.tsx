import {
  Check,
  Loader2,
  Play,
  Sliders,
  Square,
  Star,
  Trash2,
} from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import { fetchVoicePreview } from "../api";
import { describeVoice, useModelCaps } from "../model-caps";
import type { VoiceOption } from "../types";
import { voiceDetail } from "../voice-label";

// Speed slider range markers.
const SPEED_MIN = 0.75;
const SPEED_MAX = 1.5;
const SPEED_NORMAL = 1.0;
const SPEED_STEP = 0.05;

/** Clamp to the valid range and round to 2 decimal places. Without rounding, adding 0.05 with
 *  floating-point arithmetic can produce 1.0500000000000003, which looks bad in the input. */
const clampSpeed = (v: number): number => {
  if (!Number.isFinite(v)) {
    return SPEED_NORMAL;
  }
  return Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, v)) * 100) / 100;
};

const stepSpeed = (v: number, dir: number): number =>
  clampSpeed(v + dir * SPEED_STEP);

// Number of avatar color palettes in global.css (.studio-voice-avatar[data-tone]).
const VOICE_TONES = 8;

/** Hash a voice id to 0..7 so each voice consistently keeps the same avatar color. Using the
 *  character code sum as before grouped similar names into the same color. */
export const voiceColorIndex = (id: string): number => {
  let hash = 7;
  for (const char of id) {
    hash = (hash * 31 + (char.codePointAt(0) ?? 0)) % 2_147_483_647;
  }
  return hash % VOICE_TONES;
};

interface DeleteVoiceButtonProps {
  busy: boolean;
  disabled?: boolean;
  name: string;
  onDelete: () => void;
}

function DeleteVoiceButton({
  busy,
  disabled,
  name,
  onDelete,
}: DeleteVoiceButtonProps) {
  return (
    <button
      aria-label={`Delete voice ${name}`}
      className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-rose-500/10 hover:text-rose-500 disabled:opacity-40"
      disabled={disabled}
      onClick={(e) => {
        // The button is inside the voice card: clicking delete must not also select that voice.
        e.stopPropagation();
        onDelete();
      }}
      title="Delete this cloned voice (only voices from the current model can be deleted)"
      type="button"
    >
      {busy ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      ) : (
        <Trash2 className="h-3.5 w-3.5" />
      )}
    </button>
  );
}

interface PreviewVoiceButtonProps {
  disabled?: boolean;
  name: string;
  onToggle: () => void;
  state: "idle" | "loading" | "playing";
}

function PreviewVoiceButton({
  disabled,
  name,
  onToggle,
  state,
}: PreviewVoiceButtonProps) {
  let icon = <Play className="h-3.5 w-3.5 fill-current" />;
  if (state === "loading") {
    icon = <Loader2 className="h-3.5 w-3.5 animate-spin" />;
  } else if (state === "playing") {
    icon = <Square className="h-3 w-3 fill-current" />;
  }
  return (
    <button
      aria-label={
        state === "idle" ? `Preview voice ${name}` : `Stop previewing ${name}`
      }
      className={`flex h-7 w-7 items-center justify-center rounded-full transition-colors disabled:opacity-40 ${
        state === "idle"
          ? "text-muted-foreground hover:bg-secondary hover:text-brand-600 dark:hover:text-brand-400"
          : "bg-brand-500/10 text-brand-600 dark:text-brand-400"
      }`}
      disabled={disabled}
      onClick={(e) => {
        // The button is inside the voice card: previewing must not also select that voice.
        e.stopPropagation();
        onToggle();
      }}
      title={
        state === "loading"
          ? "Reading sample (the first time may take a few seconds)..."
          : "Preview this voice"
      }
      type="button"
    >
      {icon}
    </button>
  );
}

interface VoiceSelectorProps {
  /** When provided, shows a preview button for each voice. */
  apiUrl?: string;
  disabled?: boolean;
  onChangeSpeed: (speed: number) => void;
  onCloneVoice?: () => void;
  /** Deletes a saved cloned voice; omit to hide the delete button. */
  onDeleteVoice?: (voiceId: string) => Promise<void>;
  onSelectVoice: (voiceId: string) => void;
  selectedVoiceId: string;
  speed: number;
  voices: VoiceOption[];
}

export const VoiceSelector: React.FC<VoiceSelectorProps> = ({
  apiUrl,
  onCloneVoice,
  voices,
  selectedVoiceId,
  onSelectVoice,
  speed,
  onChangeSpeed,
  onDeleteVoice,
  disabled,
}) => {
  // Supertonic voices read every language: voice cards show the selected language.
  const { language, languageName, languages, setLanguage } = useModelCaps();
  // Selecting a language in "Text language" shows only Kokoro voices for that language. Voices
  // without an assigned language (VieNeu, Supertonic, cloned voices) are always shown.
  const matchesLanguage = (voice: VoiceOption) =>
    !(language && voice.language) || voice.language === language;
  // Voice and language must match. Whichever one changes, the other follows:
  // - voice changes (clicked, or preset in Settings): language follows the voice;
  // - language changes (in "Text language"): switch to the first voice for that language,
  //   otherwise clicking Read would use the previous voice.
  const lastVoiceId = useRef(selectedVoiceId);
  useEffect(() => {
    const current = voices.find((v) => v.id === selectedVoiceId);
    const voiceChanged = lastVoiceId.current !== selectedVoiceId;
    lastVoiceId.current = selectedVoiceId;
    if (!(language && current?.language) || current.language === language) {
      return;
    }
    if (voiceChanged) {
      if (languages.some((l) => l.code === current.language)) {
        setLanguage(current.language);
      }
      return;
    }
    const first = voices.find((v) => v.language === language);
    if (first) {
      onSelectVoice(first.id);
    }
  }, [
    language,
    languages,
    voices,
    selectedVoiceId,
    onSelectVoice,
    setLanguage,
  ]);
  const selectVoice = (voice: VoiceOption) => onSelectVoice(voice.id);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Preview: only one voice plays at a time.
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [previewState, setPreviewState] = useState<"loading" | "playing">(
    "loading"
  );
  const [previewError, setPreviewError] = useState<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const previewAbortRef = useRef<AbortController | null>(null);

  const stopPreview = () => {
    previewAbortRef.current?.abort();
    previewAbortRef.current = null;
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
    }
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewId(null);
  };

  // biome-ignore lint/correctness/useExhaustiveDependencies: only clean up when leaving the page
  useEffect(() => stopPreview, []);

  // Stop preview playback while main speech generation is active so the two sounds don't overlap.
  // biome-ignore lint/correctness/useExhaustiveDependencies: only react to disabled
  useEffect(() => {
    if (disabled) {
      stopPreview();
    }
  }, [disabled]);

  const togglePreview = async (voice: VoiceOption) => {
    const wasThis = previewId === voice.id;
    stopPreview();
    if (wasThis || !apiUrl) {
      return;
    }
    const controller = new AbortController();
    previewAbortRef.current = controller;
    setPreviewError(null);
    setPreviewId(voice.id);
    setPreviewState("loading");
    try {
      const blob = await fetchVoicePreview(apiUrl, voice.id, controller.signal);
      if (controller.signal.aborted) {
        return;
      }
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      previewUrlRef.current = url;
      previewAudioRef.current = audio;
      audio.onended = () => {
        if (previewAudioRef.current === audio) {
          stopPreview();
        }
      };
      await audio.play();
      setPreviewState("playing");
    } catch (e) {
      if (controller.signal.aborted) {
        return;
      }
      stopPreview();
      setPreviewError(
        `Unable to preview "${voice.name}": ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const handleDelete = async (voice: VoiceOption) => {
    if (
      !(
        onDeleteVoice &&
        // biome-ignore lint/suspicious/noAlert: confirm before a destructive action
        window.confirm(
          `Delete cloned voice "${voice.name}"?\nThis action cannot be undone.`
        )
      )
    ) {
      return;
    }
    setDeletingId(voice.id);
    setDeleteError(null);
    try {
      await onDeleteVoice(voice.id);
    } catch (e) {
      setDeleteError(
        `Unable to delete "${voice.name}": ${e instanceof Error ? e.message : String(e)}`
      );
    } finally {
      setDeletingId(null);
    }
  };
  // Extracted from the map loop to keep the component's complexity within lint limits.
  const renderActions = (voice: VoiceOption, isSelected: boolean) => (
    <div className="flex items-center gap-1.5">
      {apiUrl && (
        <PreviewVoiceButton
          disabled={disabled}
          name={voice.name}
          onToggle={() => togglePreview(voice)}
          state={previewId === voice.id ? previewState : "idle"}
        />
      )}
      {onDeleteVoice && voice.isCloned && (
        <DeleteVoiceButton
          busy={deletingId === voice.id}
          disabled={disabled || deletingId !== null}
          name={voice.name}
          onDelete={() => handleDelete(voice)}
        />
      )}
      {isSelected && (
        <div className="zoom-in-50 flex h-4.5 w-4.5 animate-in items-center justify-center rounded-full bg-brand-600 text-white shadow-xs">
          <Check className="h-3 w-3 stroke-[3]" />
        </div>
      )}
    </div>
  );

  const [filterGender, setFilterGender] = useState<
    "all" | "female" | "male" | "cloned"
  >("all");

  // The input keeps a string draft while typing. If the value were coerced to a number on every
  // keystroke, users couldn't clear it to retype, and "1." would jump to 1 mid-input.
  const [speedDraft, setSpeedDraft] = useState<string>(speed.toFixed(2));
  const [speedFocused, setSpeedFocused] = useState(false);

  // When the slider or +/- buttons change the value, the input should follow unless the user is typing.
  React.useEffect(() => {
    if (!speedFocused) {
      setSpeedDraft(speed.toFixed(2));
    }
  }, [speed, speedFocused]);

  const commitSpeedDraft = () => {
    setSpeedFocused(false);
    const parsed = Number.parseFloat(speedDraft.replace(",", "."));
    const next = Number.isFinite(parsed) ? clampSpeed(parsed) : speed;
    onChangeSpeed(next);
    setSpeedDraft(next.toFixed(2));
  };

  const filteredVoices = voices.filter((v) => {
    if (!matchesLanguage(v)) {
      return false;
    }
    if (filterGender === "cloned") {
      return v.isCloned;
    }
    if (filterGender === "female") {
      return v.gender === "female";
    }
    if (filterGender === "male") {
      return v.gender === "male";
    }
    return true;
  });

  const selectedRowRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selectedVoiceId && filterGender) {
      selectedRowRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [selectedVoiceId, filterGender]);

  return (
    <div className="studio-voice-selector flex h-full flex-col">
      <div className="studio-voice-heading">
        <h2>Voices</h2>
        {onCloneVoice && (
          <button
            className="text-primary text-xs"
            onClick={onCloneVoice}
            type="button"
          >
            + Voice Cloning
          </button>
        )}
      </div>
      <fieldset aria-label="Filter voices" className="studio-voice-filter">
        {(
          [
            { value: "all", label: "All" },
            { value: "male", label: "Male" },
            { value: "female", label: "Female" },
            { value: "cloned", label: "Mine" },
          ] as const
        ).map((filter) => (
          <button
            aria-pressed={filterGender === filter.value}
            key={filter.value}
            onClick={() => setFilterGender(filter.value)}
            type="button"
          >
            {filter.label}
          </button>
        ))}
      </fieldset>

      {deleteError && (
        <div
          className="flex items-start gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-2.5 py-1.5 text-rose-700 text-xs dark:text-rose-300"
          role="alert"
        >
          <span className="min-w-0 flex-1">{deleteError}</span>
          <button
            aria-label="Dismiss notification"
            className="font-bold"
            onClick={() => setDeleteError(null)}
            type="button"
          >
            ×
          </button>
        </div>
      )}

      {previewError && (
        <div
          className="flex items-start gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-2.5 py-1.5 text-rose-700 text-xs dark:text-rose-300"
          role="alert"
        >
          <span className="min-w-0 flex-1">{previewError}</span>
          <button
            aria-label="Dismiss notification"
            className="font-bold"
            onClick={() => setPreviewError(null)}
            type="button"
          >
            ×
          </button>
        </div>
      )}

      {/* Voice list */}
      <div className="studio-voice-list">
        {filteredVoices.length === 0 ? (
          <div className="flex h-20 flex-col items-center justify-center text-muted-foreground text-xs">
            <span>No matching voices found</span>
          </div>
        ) : (
          filteredVoices.map((voice) => {
            const isSelected = selectedVoiceId === voice.id;

            return (
              <div
                className={`studio-voice-row ${isSelected ? "studio-voice-selected" : ""} ${disabled ? "opacity-60" : ""}`}
                key={voice.id}
                ref={isSelected ? selectedRowRef : undefined}
              >
                <button
                  aria-pressed={isSelected}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  disabled={disabled}
                  onClick={() => selectVoice(voice)}
                  type="button"
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <div
                      className="studio-voice-avatar"
                      data-tone={voiceColorIndex(voice.id)}
                    >
                      {voice.name
                        .split(" ")
                        .map((part) => part[0])
                        .slice(-2)
                        .join("")}
                    </div>

                    {/* Voice information */}
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate font-semibold text-[13px] text-foreground">
                          {voice.name}
                        </span>
                        {/* Region is already shown below; a separate "Male" label can be misread as gender. */}
                        {voice.featured && (
                          <Star
                            aria-label="Voice selected by the VieNeu-TTS author"
                            className="h-3 w-3 shrink-0 fill-amber-400 text-amber-400"
                          />
                        )}
                      </div>
                      <p className="line-clamp-1 text-muted-foreground text-xs">
                        {languages.length > 0
                          ? describeVoice(voice.description ?? "", languageName)
                          : voiceDetail(voice) || "Available voice"}
                      </p>
                    </div>
                  </div>
                </button>
                {renderActions(voice, isSelected)}
              </div>
            );
          })
        )}
      </div>

      {/* Reading speed */}
      <div className="studio-speed">
        <div className="flex items-center justify-between gap-2 text-xs sm:text-sm">
          <div className="flex items-center gap-1.5 font-medium text-foreground/80">
            <Sliders className="h-3.5 w-3.5 text-brand-500" />
            <span>
              Speed ·{" "}
              {speed.toLocaleString("en-US", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
              ×
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button
              aria-label="Decrease speed"
              className="flex h-6 w-6 items-center justify-center rounded border border-border/80 bg-background font-bold text-foreground/70 text-sm leading-none transition-colors hover:bg-secondary hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
              disabled={disabled || speed <= SPEED_MIN}
              onClick={() => onChangeSpeed(stepSpeed(speed, -1))}
              type="button"
            >
              &minus;
            </button>
            <input
              className="h-6 w-14 rounded border border-border/80 bg-background px-1 text-center font-bold font-mono text-brand-600 text-xs outline-none focus:border-brand-500 disabled:opacity-40 dark:text-brand-400"
              disabled={disabled}
              max={SPEED_MAX}
              min={SPEED_MIN}
              onBlur={() => commitSpeedDraft()}
              onChange={(e) => setSpeedDraft(e.target.value)}
              onFocus={() => setSpeedFocused(true)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  commitSpeedDraft();
                  e.currentTarget.blur();
                }
              }}
              step={SPEED_STEP}
              type="number"
              value={speedDraft}
            />
            <button
              aria-label="Increase speed"
              className="flex h-6 w-6 items-center justify-center rounded border border-border/80 bg-background font-bold text-foreground/70 text-sm leading-none transition-colors hover:bg-secondary hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
              disabled={disabled || speed >= SPEED_MAX}
              onClick={() => onChangeSpeed(stepSpeed(speed, 1))}
              type="button"
            >
              +
            </button>
            <button
              className="ml-1 h-6 rounded border border-border/80 bg-background px-2 font-semibold text-muted-foreground text-xs transition-colors hover:bg-secondary hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
              disabled={disabled || speed === SPEED_NORMAL}
              onClick={() => onChangeSpeed(SPEED_NORMAL)}
              type="button"
            >
              Normal
            </button>
          </div>
        </div>

        <input
          className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-secondary accent-brand-500"
          disabled={disabled}
          max={SPEED_MAX}
          min={SPEED_MIN}
          onChange={(e) =>
            onChangeSpeed(clampSpeed(Number.parseFloat(e.target.value)))
          }
          step={SPEED_STEP}
          type="range"
          value={speed}
        />

        <div className="flex justify-between px-0.5 text-muted-foreground text-xs">
          <span>0.75× · Slow</span>
          <span>1.50× · Fast</span>
        </div>
      </div>
    </div>
  );
};
