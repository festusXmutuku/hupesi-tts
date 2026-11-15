import type React from "react";
import { useId } from "react";
import { useModelCaps } from "../model-caps";
import { InfoTip } from "./InfoTip";

// Default temperature from the VieNeu author for v3 models (original VieNeu-TTS Gradio app).
export const DEFAULT_TEMPERATURE = 0.8;

interface AdvancedGenerationProps {
  disabled?: boolean;
  onChangeTemperature: (value: number) => void;
  temperature: number;
}

/**
 * Emotion/temperature control shown directly like the author's Gradio app, default 0.8.
 * Hidden for Supertonic because that model ignores temperature.
 */
export const AdvancedGeneration: React.FC<AdvancedGenerationProps> = ({
  disabled,
  onChangeTemperature,
  temperature,
}) => {
  const multilingual = useModelCaps().languages.length > 0;
  // TTS Studio and Conversation are both in the DOM: a fixed id would collide.
  const inputId = useId();
  if (multilingual) {
    return null;
  }
  return (
    <div className="studio-advanced rounded-xl border border-border/80 bg-card px-3 py-2 shadow-sm">
      <div className="space-y-1.5">
        <div className="info-tip-row flex items-center justify-between gap-2 text-xs">
          <span className="flex items-center gap-1">
            <label className="font-medium text-foreground/80" htmlFor={inputId}>
              Emotion (Temperature)
            </label>
            <InfoTip label="Emotion">
              <strong>How much the intonation varies.</strong>
              <span>
                The VieNeu author recommends 0.7–0.8; 0.8 is the most stable default.
              </span>
              <span>
                Lower: more even, less expressive. Higher: more expressive, more likely to glitch.
              </span>
              <span>Nano models do not use this parameter.</span>
            </InfoTip>
          </span>
          <div className="flex items-center gap-1">
            <span className="font-bold font-mono text-brand-600 dark:text-brand-400">
              {temperature.toLocaleString("en-US", {
                minimumFractionDigits: 1,
              })}
            </span>
            <button
              className="ml-1 h-6 rounded border border-border/80 bg-background px-2 font-semibold text-muted-foreground text-xs transition-colors hover:bg-secondary hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
              disabled={disabled || temperature === DEFAULT_TEMPERATURE}
              onClick={() => onChangeTemperature(DEFAULT_TEMPERATURE)}
              type="button"
            >
              Default
            </button>
          </div>
        </div>
        <input
          aria-describedby={`${inputId}-hint`}
          className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-secondary accent-brand-500"
          disabled={disabled}
          id={inputId}
          max="1.5"
          min="0.1"
          onChange={(e) =>
            onChangeTemperature(Number.parseFloat(e.target.value))
          }
          step="0.1"
          type="range"
          value={temperature}
        />
        <p
          className="text-[12px] text-muted-foreground leading-relaxed"
          id={`${inputId}-hint`}
        >
          0.7 - 0.8: Most natural
        </p>
      </div>
    </div>
  );
};
