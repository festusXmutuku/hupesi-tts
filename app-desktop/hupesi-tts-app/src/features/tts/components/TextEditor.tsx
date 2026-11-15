import {
  BookOpen,
  Clipboard,
  Copy,
  Eraser,
  Sparkles,
  Trash2,
  Undo2,
} from "lucide-react";
import type React from "react";
import { useEffect, useState } from "react";
import { useModelCaps } from "../model-caps";
import { cleanMarkdown } from "../text-tools";
import { useCaretInsert } from "../use-caret-insert";
import { EmotionBar } from "./EmotionBar";

/** How long to show the notice next to the Clean up button. */
const NOTICE_MS = 3000;

interface TextEditorProps {
  disabled?: boolean;
  onChange: (val: string) => void;
  /** Open the pronunciation dictionary (KTX → dormitory...). */
  onOpenDictionary?: () => void;
  text: string;
}

const PRESET_SAMPLES = [
  {
    label: "About",
    text: "Hello everyone! Welcome to Hupesi TTS. This is a high-quality Vietnamese AI voice with natural and expressive pronunciation.",
  },
  {
    label: "Emotions",
    text: "Hi everyone! I have some really exciting news to share today. [laughs] Our project has officially been completed with great success! [sighs] But honestly, it was quite a lot of hard work.",
  },
  {
    label: "Narration",
    text: "[clears throat] Distinguished guests, today we are proud to introduce VieNeu-TTS v3 Turbo technology with real-time speech synthesis capabilities.",
  },
  {
    label: "Storytelling",
    text: "Once upon a time, on a vast meadow full of wildflowers, there lived a little fox who dreamed of soaring across the deep blue sky. [sighs] When would that dream ever come true?",
  },
  {
    label: "Advertisement",
    text: "Discover an automatic voice-over tool for short videos with crisp 48 kHz audio. [laughs] Fast, professional, and completely free!",
  },
];

// Multilingual models (Supertonic) are used for foreign-language text, so samples
// are in English (the default language) and use their own emotion tags.
const PRESET_SAMPLES_FOREIGN = [
  {
    label: "About",
    text: "Hello everyone! Welcome to Hupesi TTS. This is a high-quality AI voice that sounds natural and expressive.",
  },
  {
    label: "Emotions",
    text: "Hi everyone! I have some really exciting news to share today. <laugh> Our project has officially been completed! <sigh> But honestly, it was quite a lot of work.",
  },
  {
    label: "Narration",
    text: "Ladies and gentlemen, <breath> today we are proud to present a speech technology that runs in real time on an ordinary computer.",
  },
  {
    label: "Storytelling",
    text: "Once upon a time, on a vast meadow full of wildflowers, there lived a little fox who dreamed of flying across the deep blue sky. <sigh> When would that dream ever come true?",
  },
  {
    label: "Advertisement",
    text: "Discover an automatic voice-over tool for short videos, with crisp studio-quality sound. <laugh> Fast, professional and completely free!",
  },
];

export const TextEditor: React.FC<TextEditorProps> = ({
  text,
  onChange,
  onOpenDictionary,
  disabled,
}) => {
  const { languages, readsVietnamese } = useModelCaps();
  const multilingual = languages.length > 0;
  const charCount = text.length;
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const sentenceCount = text.trim()
    ? (text.match(/[^.!?]+[.!?]+/g) || [text]).length
    : 0;

  const handlePaste = async () => {
    try {
      const clipboardText = await navigator.clipboard.readText();
      if (clipboardText) {
        onChange(text ? `${text}\n${clipboardText}` : clipboardText);
      }
    } catch {
      // Keep the current content if the clipboard cannot be read
    }
  };

  const handleCopy = () => {
    if (text) {
      navigator.clipboard.writeText(text);
    }
  };

  const {
    textareaRef,
    onFocus,
    insertTag: handleInsertEmotion,
  } = useCaretInsert(text, onChange, disabled);
  // Undo only makes sense while the text is still exactly the cleaned version; if the
  // user edits further, the button hides itself to avoid overwriting their changes.
  const [undo, setUndo] = useState<{ before: string; after: string } | null>(
    null
  );
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!notice) {
      return;
    }
    const timer = setTimeout(() => setNotice(null), NOTICE_MS);
    return () => clearTimeout(timer);
  }, [notice]);

  const handleClean = () => {
    const cleaned = cleanMarkdown(text);
    if (cleaned === text) {
      setNotice("No formatting symbols to clean");
      return;
    }
    setUndo({ before: text, after: cleaned });
    onChange(cleaned);
    setNotice("Formatting symbols cleaned");
  };

  const canUndo = undo !== null && text === undo.after;
  const handleUndo = () => {
    if (!canUndo) {
      return;
    }
    onChange(undo.before);
    setUndo(null);
    setNotice(null);
  };

  return (
    <div className="studio-text-editor">
      <div className="studio-editor-heading">
        <h2>Text to Speech</h2>
        <span className="studio-editor-stats">
          {charCount.toLocaleString("en-US")} characters · {wordCount} words ·{" "}
          {sentenceCount} sentences · ~{Math.ceil(wordCount / 2.5)} seconds
        </span>
      </div>
      <div className="studio-editor-toolbar">
        {/* Text tools */}
        <div className="flex items-center gap-1.5">
          {notice && (
            <span className="hidden text-muted-foreground text-xs italic md:inline">
              {notice}
            </span>
          )}
          {canUndo && (
            <button
              className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-amber-600 text-xs transition-colors hover:bg-amber-500/10 disabled:opacity-50 dark:text-amber-400"
              disabled={disabled}
              onClick={handleUndo}
              title="Restore the text before cleaning"
              type="button"
            >
              <Undo2 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Undo</span>
            </button>
          )}
          <button
            className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-muted-foreground text-xs transition-colors hover:bg-secondary/80 hover:text-foreground disabled:opacity-50"
            disabled={!text || disabled}
            onClick={handleClean}
            title="Remove formatting symbols when pasting from ChatGPT/Gemini… (**bold**, ## headings, > quotes, links) so the app doesn't read them aloud"
            type="button"
          >
            <Eraser className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Clean up</span>
          </button>
          {/* Pronunciation dictionary follows Vietnamese reading rules: hide when the model can't read Vietnamese (Kokoro). */}
          {onOpenDictionary && readsVietnamese && (
            <button
              className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-violet-600 text-xs transition-colors hover:bg-violet-500/10 dark:text-violet-400"
              onClick={onOpenDictionary}
              title="Pronunciation Dictionary - teach the app how to read abbreviations (KTX → dormitory)"
              type="button"
            >
              <BookOpen className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Dictionary</span>
            </button>
          )}
          <button
            className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-muted-foreground text-xs transition-colors hover:bg-secondary/80 hover:text-foreground disabled:opacity-50"
            disabled={disabled}
            onClick={handlePaste}
            title="Paste from clipboard"
            type="button"
          >
            <Clipboard className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Paste</span>
          </button>
          <button
            className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-muted-foreground text-xs transition-colors hover:bg-secondary/80 hover:text-foreground disabled:opacity-50"
            disabled={!text || disabled}
            onClick={handleCopy}
            title="Copy text"
            type="button"
          >
            <Copy className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Copy</span>
          </button>
          <button
            className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-rose-500 text-xs transition-colors hover:bg-rose-500/10 disabled:opacity-50"
            disabled={!text || disabled}
            onClick={() => onChange("")}
            title="Clear all"
            type="button"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Delete</span>
          </button>
        </div>
      </div>

      <div className="studio-emotions">
        <EmotionBar disabled={disabled} onInsertTag={handleInsertEmotion} />
      </div>
      {/* The text area scrolls independently without scrolling the page. */}
      <div className="studio-textarea-area">
        <textarea
          aria-label="Text to read"
          className="studio-textarea"
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          onFocus={onFocus}
          placeholder={`Enter or paste the ${multilingual ? "" : "Vietnamese "}text you want to convert to speech here...`}
          ref={textareaRef}
          spellCheck={false}
          value={text}
        />
      </div>

      {/* Preset text samples */}
      <div className="flex flex-wrap items-center gap-1.5 border-border/40 border-t bg-muted/20 px-3 py-1.5 sm:px-3.5">
        <span className="mr-0.5 flex items-center gap-1 font-medium text-muted-foreground text-xs">
          <Sparkles className="h-3.5 w-3.5 text-amber-500" />
          Samples:
        </span>
        {(multilingual ? PRESET_SAMPLES_FOREIGN : PRESET_SAMPLES).map(
          (sample) => (
            <button
              className="rounded-full border border-border/60 bg-secondary/80 px-2.5 py-0.5 font-medium text-foreground/80 text-xs transition-all hover:border-brand-500/40 hover:bg-secondary hover:text-foreground active:scale-95"
              disabled={disabled}
              key={sample.label}
              onClick={() => onChange(sample.text)}
              type="button"
            >
              {sample.label}
            </button>
          )
        )}
      </div>
    </div>
  );
};
