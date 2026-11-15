import { Laugh, Megaphone, Pause, Sparkles, Wind } from "lucide-react";
import type React from "react";
import { useModelCaps } from "../model-caps";

interface EmotionBarProps {
  disabled?: boolean;
  onInsertTag: (tag: string) => void;
}

const EMOTIONS = [
  {
    tag: "[c\u01b0\u1eddi]",
    label: "Laugh",
    icon: Laugh,
    color:
      "text-amber-800 dark:text-amber-400 hover:bg-amber-500/25 border-amber-500/45 hover:border-amber-500/70 bg-amber-500/15",
  },
  {
    tag: "[h\u1eafng gi\u1ecdng]",
    label: "Throat clear",
    icon: Megaphone,
    color:
      "text-purple-800 dark:text-purple-300 hover:bg-purple-500/25 border-purple-500/45 hover:border-purple-500/70 bg-purple-500/15",
  },
  {
    tag: "[th\u1edf d\u00e0i]",
    label: "Sigh",
    icon: Wind,
    color:
      "text-sky-800 dark:text-sky-300 hover:bg-sky-500/25 border-sky-500/45 hover:border-sky-500/70 bg-sky-500/15",
  },
];

// Supertonic (multilingual model) tags: inserted as-is, the server keeps them unchanged.
const SUPERTONIC_EMOTIONS = [
  { ...EMOTIONS[0], tag: "<laugh>" },
  {
    tag: "<breath>",
    label: "Breath",
    icon: Wind,
    color:
      "text-emerald-800 dark:text-emerald-300 hover:bg-emerald-500/25 border-emerald-500/45 hover:border-emerald-500/70 bg-emerald-500/15",
  },
  { ...EMOTIONS[2], tag: "<sigh>" },
];

const PAUSE_TAG = "[ngh\u1ec9 1s]";

export const EmotionBar: React.FC<EmotionBarProps> = ({
  onInsertTag,
  disabled,
}) => {
  // Multilingual models (Supertonic) use their own tags: <laugh> <breath> <sigh>.
  const { languages, supportsEmotions } = useModelCaps();
  const multilingual = languages.length > 0;
  // Kokoro has no emotion tags: only the pause button remains.
  let emotions = multilingual ? SUPERTONIC_EMOTIONS : EMOTIONS;
  if (!supportsEmotions) {
    emotions = [];
  }
  return (
    <div className="flex flex-wrap items-center justify-between gap-1.5 rounded-lg border border-border/60 bg-secondary/25 px-2.5 py-1 sm:py-1.5">
      <div className="flex items-center gap-1.5 font-medium text-foreground/90 text-xs sm:text-[13px]">
        <Sparkles
          aria-hidden="true"
          className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400"
        />
        <span>Emotions:</span>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {emotions.map((item) => (
          <button
            className={`inline-flex select-none items-center gap-1.5 rounded-md border px-2.5 py-1 font-semibold text-xs shadow-xs transition-all active:scale-95 sm:text-[13px] ${
              item.color
            } ${disabled ? "cursor-not-allowed opacity-40" : "cursor-pointer"}`}
            disabled={disabled}
            key={item.tag}
            onClick={() => onInsertTag(item.tag)}
            title={`Insert emotion tag ${item.tag}`}
            type="button"
          >
            <item.icon aria-hidden="true" className="h-3.5 w-3.5" />
            <span>{item.label}</span>
            <code className="ml-0.5 font-mono text-[12px] opacity-75 sm:text-[12px]">
              {item.tag}
            </code>
          </button>
        ))}
        {/* Pause tag: the server splits text at the tag and inserts the specified silence.
            Works with all models, not just v3 Turbo. Edit the duration after inserting. */}
        <button
          className={`inline-flex select-none items-center gap-1.5 rounded-md border border-slate-500/45 bg-slate-500/15 px-2.5 py-1 font-semibold text-slate-600 text-xs shadow-xs transition-all hover:border-slate-500/70 hover:bg-slate-500/25 active:scale-95 sm:text-[13px] dark:text-slate-300 ${disabled ? "cursor-not-allowed opacity-40" : "cursor-pointer"}`}
          disabled={disabled}
          onClick={() => onInsertTag(PAUSE_TAG)}
          title={`Insert a pause. Edit the duration in the tag: ${PAUSE_TAG.replace("1s", "2s")}, [ngh\u1ec9 0.5s], [ngh\u1ec9 500ms] - max 10 seconds.`}
          type="button"
        >
          <Pause className="h-3.5 w-3.5" />
          <span>Pause</span>
          <code className="ml-0.5 font-mono text-[12px] opacity-75 sm:text-[12px]">
            {PAUSE_TAG}
          </code>
        </button>
      </div>
    </div>
  );
};
