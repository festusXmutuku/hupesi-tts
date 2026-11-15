import {
  AlertTriangle,
  Code2,
  Cpu,
  ExternalLink,
  Globe,
  Lock,
  Sparkles,
  Volume2,
  Zap,
} from "lucide-react";
import type React from "react";
import type { AppUpdateInfo } from "../api";
import { UpdateCheckButton } from "./AppUpdate";

// Hupesi TTS source repository (used in both source and footer links).
const REPO_CIT = "https://github.com/festusxmutukugithub/hupesi-tts";

interface AboutTabProps {
  apiUrl: string;
  engineName?: string;
  onUpdateChecked: (info: AppUpdateInfo) => void;
  sampleRate?: number;
  /** Current version + latest version (from /api/app/update); null when not yet fetched. */
  update: AppUpdateInfo | null;
}

export const AboutTab: React.FC<AboutTabProps> = ({
  apiUrl,
  onUpdateChecked,
  update,
  engineName = "VieNeu-TTS v3 Turbo",
  sampleRate = 48_000,
}) => {
  return (
    <div className="fade-in-50 mx-auto w-full animate-in space-y-6 pb-10 duration-300">
      {/* 1. Hero Banner */}
      <div className="relative overflow-hidden rounded-[14px] border border-border bg-gradient-to-br from-card via-card/95 to-card p-6 shadow-md md:p-8">
        <div className="pointer-events-none absolute top-0 right-0 -mt-8 -mr-8 h-56 w-56 rounded-full bg-brand-500/10 blur-3xl" />
        <div className="pointer-events-none absolute bottom-0 left-1/3 -mb-12 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl" />

        <div className="relative flex flex-col items-start justify-between gap-6 md:flex-row md:items-center">
          <div className="flex items-start gap-4">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-tr from-brand-500 via-teal-500 to-cyan-500 text-white shadow-brand-500/25 shadow-md">
              <Volume2 className="h-7 w-7" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2.5">
                <h2 className="font-black text-2xl text-foreground tracking-tight md:text-3xl">
                  Hupesi TTS
                </h2>
                <span className="rounded-full border border-brand-500/30 bg-brand-500/15 px-2.5 py-0.5 font-bold text-brand-600 text-xs uppercase shadow-xs dark:text-brand-400">
                  Desktop Edition{update ? ` v${update.current}` : ""}
                </span>
              </div>
              <p className="mt-1.5 max-w-2xl text-muted-foreground text-sm leading-relaxed">
                An AI solution for high-quality Vietnamese text-to-speech (TTS)
                and voice cloning, processed entirely on-device and offline.
              </p>
              <div className="mt-3">
                <UpdateCheckButton
                  apiUrl={apiUrl}
                  info={update}
                  onChecked={onUpdateChecked}
                />
              </div>
            </div>
          </div>

          {/* Quick Stats Badges */}
          <div className="flex w-full shrink-0 flex-row gap-2 md:w-auto md:flex-col">
            <div className="flex flex-1 items-center gap-2 rounded-xl border border-border/60 bg-secondary/60 px-3.5 py-2 text-xs md:flex-none">
              <Cpu className="h-4 w-4 text-brand-500" />
              <div>
                <div className="font-semibold text-[12px] text-muted-foreground uppercase">
                  Models AI
                </div>
                <div className="font-bold text-foreground">{engineName}</div>
              </div>
            </div>
            <div className="flex flex-1 items-center gap-2 rounded-xl border border-border/60 bg-secondary/60 px-3.5 py-2 text-xs md:flex-none">
              <Zap className="h-4 w-4 text-cyan-500" />
              <div>
                <div className="font-semibold text-[12px] text-muted-foreground uppercase">
                  Audio
                </div>
                <div className="font-bold text-foreground">
                  {sampleRate / 1000} kHz Hi-Fi
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Key Highlights Cards */}
      <div className="studio-feature-grid">
        <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-card p-4 shadow-xs">
          <div className="shrink-0 rounded-lg bg-brand-500/10 p-2.5 text-brand-500">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-bold text-foreground text-xs uppercase tracking-wide">
              {sampleRate / 1000} kHz Hi-Fi Quality
            </h4>
            <p className="mt-1 text-muted-foreground text-xs leading-relaxed">
              Natural, clear voices powered by VieNeu-Codec technology.
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-card p-4 shadow-xs">
          <div className="shrink-0 rounded-lg bg-cyan-500/10 p-2.5 text-cyan-500">
            <Lock className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-bold text-foreground text-xs uppercase tracking-wide">
              100% Local & Private
            </h4>
            <p className="mt-1 text-muted-foreground text-xs leading-relaxed">
              Runs on-device on your computer and never sends text or audio
              outside.
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-card p-4 shadow-xs">
          <div className="shrink-0 rounded-lg bg-indigo-500/10 p-2.5 text-indigo-500">
            <Zap className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-bold text-foreground text-xs uppercase tracking-wide">
              Your Voice Cloning
            </h4>
            <p className="mt-1 text-muted-foreground text-xs leading-relaxed">
              Record from a microphone or import audio files (MP3, WAV, M4A)
              that you have permission to use. Voices are stored only on your
              machine.
            </p>
          </div>
        </div>
      </div>

      {/* 3. Usage warning: voice cloning and impersonation are the biggest risks for an offline TTS app. */}
      <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-5 shadow-xs">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
          <h3 className="font-bold text-amber-700 text-sm uppercase tracking-wide dark:text-amber-300">
            Warning
          </h3>
        </div>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-foreground/90 text-sm leading-relaxed">
          <li>
            <strong>Voice permission required:</strong> only clone your own
            voice or a voice you have explicit permission to use. The app does
            not verify identity or proof of ownership.
          </li>
          <li>
            <strong>No impersonation:</strong> do not mimic other people
            (family members, celebrities, officials, organizations, or
            businesses) to scam, solicit money, defame, insult, or mislead.
          </li>
          <li>
            <strong>Respect copyright:</strong> text content (books, stories,
            articles) must be yours or used with permission.
          </li>
          <li>
            When publishing content that simulates a real person&apos;s voice,{" "}
            <strong>clearly disclose that it was created with AI</strong>.
          </li>
        </ul>
        <p className="mt-3 border-amber-500/30 border-t pt-2 text-muted-foreground text-xs leading-relaxed">
          Users are responsible for the content they create. The authors of CIT
          Voice Studio, VieNeu-TTS, Supertone, and Kokoro are not responsible
          for misuse.
        </p>
      </div>

      {/* 4. Attribution */}
      <div className="space-y-3 rounded-xl border border-border/80 bg-card p-5 shadow-xs">
        <p className="text-foreground/90 text-sm leading-relaxed">
          <strong>Hupesi TTS</strong> is developed by{" "}
          <strong>Festus Mutuku</strong>, based on <strong>VieNeu-TTS</strong> by{" "}
          <strong>Pham Nguyen Ngoc Bao</strong> (Vietnamese voices).
          Non-Vietnamese voices use <strong>Supertonic 3</strong> by{" "}
          <strong>Supertone Inc.</strong> and <strong>Kokoro-82M</strong> by{" "}
          <strong>hexgrad</strong>, downloaded separately when selected. This is
          an independent application and is not an official product of
          VieNeu-TTS, Supertone, or Kokoro.
        </p>

        {/* Clearly identify the owner of each source repository. */}
        <div className="space-y-2 pt-1">
          <div className="flex flex-wrap items-baseline gap-2 text-xs">
            <span className="w-28 shrink-0 text-muted-foreground">
              Hupesi TTS
            </span>
            <a
              className="inline-flex items-center gap-1 font-bold text-sky-500 transition-colors hover:text-sky-400"
              href={REPO_CIT}
              rel="noreferrer"
              target="_blank"
            >
              <Code2 className="h-3.5 w-3.5" />
              <span>festusxmutukugithub/hupesi-tts</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
            <a
              className="inline-flex items-center gap-1 font-bold text-sky-500 transition-colors hover:text-sky-400"
              href="https://github.com/festusxmutukugithub"
              rel="noreferrer"
              target="_blank"
            >
              <Globe className="h-3.5 w-3.5" />
              <span>github.com/festusxmutukugithub</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
          </div>

          <div className="flex flex-wrap items-baseline gap-2 text-xs">
            <span className="w-28 shrink-0 text-muted-foreground">
              VieNeu-TTS (upstream)
            </span>
            <a
              className="inline-flex items-center gap-1 font-bold text-brand-600 hover:underline dark:text-brand-400"
              href="https://github.com/pnnbao97/VieNeu-TTS"
              rel="noreferrer"
              target="_blank"
            >
              <Code2 className="h-3.5 w-3.5" />
              <span>pnnbao97/VieNeu-TTS</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
            <a
              className="inline-flex items-center gap-1 font-bold text-amber-600 hover:underline dark:text-amber-400"
              href="https://huggingface.co/pnnbao-ump/VieNeu-TTS"
              rel="noreferrer"
              target="_blank"
            >
              <span>Hugging Face</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
          </div>

          <div className="flex flex-wrap items-baseline gap-2 text-xs">
            <span className="w-28 shrink-0 text-muted-foreground">
              Supertonic 3
            </span>
            <a
              className="inline-flex items-center gap-1 font-bold text-brand-600 hover:underline dark:text-brand-400"
              href="https://github.com/supertone-oss-archive/supertonic"
              rel="noreferrer"
              target="_blank"
            >
              <Code2 className="h-3.5 w-3.5" />
              <span>supertone-oss-archive/supertonic</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
            <a
              className="inline-flex items-center gap-1 font-bold text-amber-600 hover:underline dark:text-amber-400"
              href="https://huggingface.co/supertone-oss-archive/supertonic-3"
              rel="noreferrer"
              target="_blank"
            >
              <span>Hugging Face</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
          </div>

          <div className="flex flex-wrap items-baseline gap-2 text-xs">
            <span className="w-28 shrink-0 text-muted-foreground">
              Kokoro-82M
            </span>
            <a
              className="inline-flex items-center gap-1 font-bold text-brand-600 hover:underline dark:text-brand-400"
              href="https://github.com/hexgrad/kokoro"
              rel="noreferrer"
              target="_blank"
            >
              <Code2 className="h-3.5 w-3.5" />
              <span>hexgrad/kokoro</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
            <a
              className="inline-flex items-center gap-1 font-bold text-amber-600 hover:underline dark:text-amber-400"
              href="https://huggingface.co/hexgrad/Kokoro-82M"
              rel="noreferrer"
              target="_blank"
            >
              <span>Hugging Face</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
          </div>
        </div>

        {/* Apache 2.0 requires license notices in distributed releases (LICENSE/NOTICE files), not full display in the UI. */}
        <p className="border-border/50 border-t pt-1 text-[12px] text-muted-foreground">
          App: open source, Apache 2.0 · VieNeu-TTS: Apache 2.0 · Supertonic 3:
          OpenRAIL-M (prohibits impersonation, fraud, and undisclosed AI
          content) · Kokoro-82M: Apache 2.0 (optional audio separation add-on
          includes GPL-3.0 components, not included in the installer).
        </p>
      </div>

      {/* 5. Footer */}
      <div className="pt-1 text-center">
        <a
          className="inline-flex items-center gap-1.5 font-bold text-muted-foreground text-xs transition-colors hover:text-sky-500"
          href={REPO_CIT}
          rel="noreferrer"
          target="_blank"
        >
          <span>Festus Mutuku</span>
          <ExternalLink className="h-3 w-3 opacity-70" />
        </a>
      </div>
    </div>
  );
};
