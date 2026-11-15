import { Languages } from "lucide-react";
import type React from "react";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { type LanguageOption, setRequestLanguage } from "./api";

// Capabilities of the running model, taken from /health: which languages it can read
// (Supertonic, Kokoro), and whether it supports voice cloning and emotion tags. All tabs
// share one language selection; api.ts automatically includes it in read requests.

interface ModelCaps {
  language: string;
  /** Display name of the selected language shown on voice cards ("English"). */
  languageName: string;
  languages: LanguageOption[];
  /** Whether Vietnamese can be read (VieNeu, Supertonic: yes; Kokoro: no). The
   * pronunciation dictionary follows Vietnamese reading rules, so it only applies when available. */
  readsVietnamese: boolean;
  setLanguage: (code: string) => void;
  supportsCloning: boolean;
  supportsEmotions: boolean;
}

const DEFAULT_LANGUAGE = "en";
const MULTILINGUAL = /Multilingual|\u0110a ng\u00f4n ng\u1eef/;

/** Voice description: "Female · Multilingual · …" -> "Female · English · …" based on the selected language. */
export function describeVoice(
  description: string,
  languageName: string
): string {
  return languageName
    ? description.replace(MULTILINGUAL, languageName)
    : description;
}

const ModelCapsContext = createContext<ModelCaps>({
  language: "",
  languageName: "",
  languages: [],
  readsVietnamese: true,
  setLanguage: () => undefined,
  supportsCloning: true,
  supportsEmotions: true,
});

export function useModelCaps(): ModelCaps {
  return useContext(ModelCapsContext);
}

export const ModelCapsProvider: React.FC<{
  children: React.ReactNode;
  languages: LanguageOption[];
  supportsCloning: boolean;
  supportsEmotions: boolean;
}> = ({ children, languages, supportsCloning, supportsEmotions }) => {
  // Default to English: multilingual models are typically used for foreign-language text.
  // "" = auto-detect from text.
  const [language, setLanguage] = useState(DEFAULT_LANGUAGE);

  useEffect(() => {
    const valid = languages.some((l) => l.code === language);
    setRequestLanguage(valid ? language : null);
    if (languages.length > 0 && language && !valid) {
      setLanguage(DEFAULT_LANGUAGE);
    }
  }, [language, languages]);

  const languageName =
    languages.find((l) => l.code === language)?.name ?? "Auto-detect language";
  const readsVietnamese =
    languages.length === 0 || languages.some((l) => l.code === "vi");
  const value = useMemo(
    () => ({
      language,
      languageName,
      languages,
      readsVietnamese,
      setLanguage,
      supportsCloning,
      supportsEmotions,
    }),
    [
      language,
      languageName,
      languages,
      readsVietnamese,
      supportsCloning,
      supportsEmotions,
    ]
  );
  return (
    <ModelCapsContext.Provider value={value}>
      {children}
    </ModelCapsContext.Provider>
  );
};

/** Language selection control; hidden when the model only supports Vietnamese (VieNeu). */
export const LanguageSelect: React.FC<{
  className?: string;
  disabled?: boolean;
}> = ({ className, disabled }) => {
  const { language, languages, setLanguage } = useModelCaps();
  if (languages.length === 0) {
    return null;
  }
  return (
    <label
      className={`flex items-center gap-2 text-xs ${className ?? ""}`}
      title="Select the correct language of the text for accurate native pronunciation (this does not translate the text). The preview button also reads the sample sentence in this language. Auto-detect: reads as Vietnamese if it contains Vietnamese diacritics, as Japanese/Korean for Japanese/Korean characters, and as English otherwise."
    >
      <Languages className="h-4 w-4 shrink-0 text-brand-500" />
      <span className="shrink-0 font-semibold text-foreground/90">
        Text language
      </span>
      <select
        className="w-full min-w-0 rounded-lg border border-border/80 bg-background px-2.5 py-1.5 outline-none focus:border-brand-500"
        disabled={disabled}
        onChange={(e) => setLanguage(e.target.value)}
        value={language}
      >
        <option value="">Auto-detect from text</option>
        {languages.map((l) => (
          <option key={l.code} value={l.code}>
            {l.name}
          </option>
        ))}
      </select>
    </label>
  );
};
