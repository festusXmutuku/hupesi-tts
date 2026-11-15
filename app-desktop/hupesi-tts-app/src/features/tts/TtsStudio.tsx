import {
  AlertCircle,
  BookOpen,
  FileText,
  History,
  Info,
  ListOrdered,
  Mic,
  Settings,
  Sparkles,
  Users,
  X,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import env from "@/config/env";
import {
  type AppUpdateInfo,
  checkAppUpdate,
  checkServerHealth,
  clearHistory,
  convertToMp3,
  deleteCustomVoice,
  deleteHistory,
  fetchHistory,
  fetchUiPrefs,
  fetchVoices,
  generateToHistory,
  type LanguageOption,
  revealHistory,
  saveHistory,
  saveUiPrefs,
  type UiPrefs,
} from "./api";
import { downloadBlobAsFile, StreamingAudioEngine } from "./audio-engine";
import { AboutTab } from "./components/AboutTab";
import {
  AdvancedGeneration,
  DEFAULT_TEMPERATURE,
} from "./components/AdvancedGeneration";
import {
  SidebarFooter,
  UpdateBanner,
  UpdateDialog,
} from "./components/AppUpdate";
import { AudioPlayerBar } from "./components/AudioPlayerBar";
import { BatchTab } from "./components/BatchTab";
import { ConversationTab } from "./components/ConversationTab";
import { Header } from "./components/Header";
import { HistoryList } from "./components/HistoryList";
import { HistoryStorageNote } from "./components/HistoryStorageNote";
import { InfoTip } from "./components/InfoTip";
import { LoadingScreen } from "./components/LoadingScreen";
import { PdfReaderTab } from "./components/PdfReaderTab";
import { PronunciationDictionaryModal } from "./components/PronunciationDictionary";
import { SavedFileToast } from "./components/SavedFileToast";
import { SettingsModal } from "./components/SettingsModal";
import { TermsGate } from "./components/TermsGate";
import { TextEditor } from "./components/TextEditor";
import { VoiceCloningTab } from "./components/VoiceCloningTab";
import { VoiceSelector } from "./components/VoiceSelector";
import {
  downloadHistoryAudio,
  entryBlob,
  entryFileBase,
  entrySubtitles,
  fromServer,
  srtBlob,
} from "./history";
import { LanguageSelect, ModelCapsProvider } from "./model-caps";
import { pauseOtherAudio, toggleAudioPlayback } from "./playback";
import { startStartupPolling } from "./startup-polling";
import type {
  ActiveTab,
  AppSettings,
  ConnectionStatus,
  HistoryEntry,
  PlaybackStatus,
  VoiceOption,
} from "./types";

const getInitialApiUrl = (): string => {
  // Vite dev (port 1420, opened via localhost or 127.0.0.1) has no backend - previously it only
  // excluded "http://localhost:1420", so opening via 127.0.0.1 got stuck on the loading screen.
  if (
    !import.meta.env.DEV &&
    typeof window !== "undefined" &&
    window.location.origin &&
    !window.location.origin.startsWith("file:")
  ) {
    return window.location.origin;
  }
  return env.API_URL || "http://127.0.0.1:8001";
};

const DEFAULT_SETTINGS: AppSettings = {
  apiUrl: getInitialApiUrl(),
  playbackSpeed: 1.0,
  autoPlayOnGenerate: true,
  sampleRate: 48_000,
  silenceDuration: 0.2,
};

const DEFAULT_TEXT =
  "Hello everyone! Welcome to Hupesi TTS. This is an AI text-to-speech solution for Vietnamese with studio-quality 48kHz audio, supporting voice cloning, multi-speaker podcasts, and document/PDF reading.";
// Doan mau khi dang dung mo hinh giong nuoc ngoai (Supertonic, mac dinh tieng Anh).
const DEFAULT_TEXT_FOREIGN =
  "Hello and welcome to Hupesi TTS. This voice runs entirely on your computer, with no internet connection needed. Choose a language and a voice, then press the button to listen.";

// Seed used when "Stable" is enabled. Fixed (not random) so reopening the app produces
// the same result; changing this value changes the voice of all future "Stable" generations.
const STABLE_SEED = 20_251_021;

export const TtsStudio: React.FC = () => {
  useEffect(() => {
    const frame = requestAnimationFrame(() =>
      document.getElementById("bootstrap-screen")?.remove()
    );
    return () => cancelAnimationFrame(frame);
  }, []);
  // Navigation and tabs
  const [activeTab, setActiveTab] = useState<ActiveTab>("tts");
  const [restoredConversation, setRestoredConversation] = useState<{
    key: number;
    script: string;
  } | null>(null);

  // Settings and light/dark theme
  const [settings, setSettings] = useState<AppSettings>(() => {
    const saved = localStorage.getItem("hupesi_tts_settings");
    return saved
      ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) }
      : DEFAULT_SETTINGS;
  });

  const [isDark, setIsDark] = useState<boolean>(() => {
    return (
      localStorage.getItem("hupesi_tts_theme") === "dark" ||
      window.matchMedia("(prefers-color-scheme: dark)").matches
    );
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [settingsSection, setSettingsSection] = useState<
    "models" | undefined
  >();
  const [isDictionaryOpen, setIsDictionaryOpen] = useState(false);

  // Server status and startup screen
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [engineName, setEngineName] = useState<string>("VieNeu-TTS v3 Turbo");
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(true);
  const [loadingMessage, setLoadingMessage] = useState<string>(
    "Starting backend and loading model..."
  );
  const [loadingState, setLoadingState] = useState<
    "starting" | "downloading" | "ready" | "failed"
  >("starting");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Phien ban + ban moi: hoi mot lan khi server san sang (server tu gioi han 1 lan/ngay).
  const [appUpdate, setAppUpdate] = useState<AppUpdateInfo | null>(null);
  const [updateDismissed, setUpdateDismissed] = useState(false);
  const [updateDialogOpen, setUpdateDialogOpen] = useState(false);
  const closeUpdateDialog = useCallback(() => setUpdateDialogOpen(false), []);
  // User-initiated check: if a new version is available, always open the dialog, even if previously postponed.
  const handleUpdateChecked = useCallback((info: AppUpdateInfo) => {
    setAppUpdate(info);
    setUpdateDismissed(false);
    if (info.newer) {
      setUpdateDialogOpen(true);
    }
  }, []);

  // Text-to-speech data
  const [voices, setVoices] = useState<VoiceOption[]>([]);
  // Kha nang cua mo hinh dang chay (tu /health): ngon ngu doc duoc, co nhan ban khong.
  const [modelLanguages, setModelLanguages] = useState<LanguageOption[]>([]);
  const [supportsCloning, setSupportsCloning] = useState(true);
  const [supportsEmotions, setSupportsEmotions] = useState(true);
  const [selectedVoiceId, setSelectedVoiceId] = useState<string>("");
  // Track the applied default voice so changing the voice list (after cloning, etc.) doesn't pull
  // the user's selection back to it. Reset when the model changes so it can be reapplied.
  const appliedDefaultVoice = useRef("");
  const [text, setText] = useState<string>(DEFAULT_TEXT);
  const [speed, setSpeed] = useState<number>(1.0);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [volume, setVolume] = useState<number>(1.0);

  // Audio and playback state
  const [playbackStatus, setPlaybackStatus] = useState<PlaybackStatus>("idle");
  const [ttfaMs, setTtfaMs] = useState<number | null>(null);
  const [genProgress, setGenProgress] = useState<{
    elapsedMs: number;
    percent: number;
  } | null>(null);
  const [currentAudioBlobUrl, setCurrentAudioBlobUrl] = useState<string | null>(
    null
  );
  const [currentAudioBlob, setCurrentAudioBlob] = useState<Blob | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  // Tao kem phu de .srt (doc theo tung cau de do moc thoi gian that).
  const [withSubtitles, setWithSubtitles] = useState(false);
  // "Stable": fixed seed so the same text + voice produces the exact same result, even after
  // reopening the app. Off by default to keep the previous behavior (different intonation each time).
  const [stableVoice, setStableVoice] = useState(false);
  // Kokoro always reads the same way: hide "Stable" and don't send a seed.
  const [supportsSeed, setSupportsSeed] = useState(true);
  const [currentSubtitles, setCurrentSubtitles] = useState<string | null>(null);
  // Muc dang phat: chan ket qua tai cham (blob/SRT cua muc cu) ghi de len muc moi.
  const currentEntryIdRef = useRef<string | null>(null);
  const [currentAudioEntry, setCurrentAudioEntry] =
    useState<HistoryEntry | null>(null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const sharedAudioRef = useRef<HTMLAudioElement | null>(null);

  // Refs for player and in-flight requests
  const audioEngineRef = useRef<StreamingAudioEngine>(
    new StreamingAudioEngine(48_000)
  );
  const abortControllerRef = useRef<AbortController | null>(null);
  const genTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(
    () => () => {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      if (genTimerRef.current) {
        clearInterval(genTimerRef.current);
        genTimerRef.current = null;
      }
    },
    []
  );

  // Sync dark mode class on the HTML document
  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add("dark");
      localStorage.setItem("hupesi_tts_theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("hupesi_tts_theme", "light");
    }
  }, [isDark]);

  // UI preferences saved on the server. The desktop app (PyWebView private mode)
  // clears localStorage on exit, so localStorage is only a temporary value at startup; the
  // file on disk is the source of truth. Read only once when the app opens.
  const prefsApiUrl = useRef(settings.apiUrl);
  useEffect(() => {
    const controller = new AbortController();
    fetchUiPrefs(prefsApiUrl.current, controller.signal)
      .then((prefs) => {
        if (prefs.theme) {
          setIsDark(prefs.theme === "dark");
        }
        const { theme: _theme, ...rest } = prefs;
        if (Object.keys(rest).length > 0) {
          setSettings((prev) => {
            const next = { ...prev, ...rest };
            localStorage.setItem("hupesi_tts_settings", JSON.stringify(next));
            return next;
          });
        }
      })
      .catch(() => {
        // Old/unavailable server: keep the value in localStorage.
      });
    return () => controller.abort();
  }, []);

  const persistUiPrefs = useCallback(
    (prefs: UiPrefs) => {
      saveUiPrefs(settings.apiUrl, prefs).catch(() => {
        // If it can't be saved to disk, localStorage still works for this session.
      });
    },
    [settings.apiUrl]
  );

  // Preferences and the voice list may arrive in any order, so wait for both. If the voice
  // is not available in this model, keep the model's default voice.
  useEffect(() => {
    const voiceId = settings.defaultVoiceId;
    if (
      voiceId &&
      voiceId !== appliedDefaultVoice.current &&
      voices.some((voice) => voice.id === voiceId)
    ) {
      appliedDefaultVoice.current = voiceId;
      setSelectedVoiceId(voiceId);
    }
  }, [settings.defaultVoiceId, voices]);

  // Add to history
  const handleAddHistory = (entry: HistoryEntry) => {
    setHistory((prev) => [entry, ...prev]);
    currentEntryIdRef.current = entry.id;
    setCurrentAudioEntry(entry);
    setCurrentAudioBlobUrl(entry.audioBlobUrl);
    setCurrentAudioBlob(entry.audioBlob ?? null);
    setCurrentSubtitles(entry.subtitles ?? null);
    if (!entry.audioBlob) {
      return;
    }
    // Luu xuong o dia qua server de con lai sau khi tat app. Loi thi van giu trong
    // phien hien tai, chi mat khi tat app.
    saveHistory(settings.apiUrl, {
      audio: entry.audioBlob,
      durationSeconds: entry.durationSeconds,
      format: entry.format ?? "wav",
      mode: entry.mode ?? "tts",
      subtitles: entry.subtitles,
      text: entry.text,
      voiceId: entry.voiceId,
      voiceName: entry.voiceName,
    })
      .then((saved) =>
        setHistory((prev) =>
          prev.map((e) =>
            e.id === entry.id ? { ...e, serverId: saved.id } : e
          )
        )
      )
      .catch((e) => console.warn("[History] Could not save history:", e));
  };

  const loadHistory = useCallback(async (apiUrl: string) => {
    try {
      const saved = (await fetchHistory(apiUrl)).map((e) =>
        fromServer(apiUrl, e)
      );
      // Giu muc chua luu kip (dang luu / luu loi) o dau danh sach.
      setHistory((prev) => [...prev.filter((e) => !e.serverId), ...saved]);
    } catch (e) {
      console.warn("[History] Could not read saved history:", e);
    }
  }, []);

  const handlePlayHistory = (entry: HistoryEntry) => {
    const audio = sharedAudioRef.current;
    if (audio && currentAudioBlobUrl === entry.audioBlobUrl) {
      void toggleAudioPlayback(audio);
      return;
    }
    audio?.pause();
    setIsAudioPlaying(false);
    currentEntryIdRef.current = entry.id;
    setCurrentAudioEntry(entry);
    setCurrentAudioBlobUrl(entry.audioBlobUrl);
    setCurrentAudioBlob(entry.audioBlob ?? null);
    setCurrentSubtitles(entry.subtitles ?? null);
    setTtfaMs(null);
    // Muc nap tu o dia: KHONG tai ca khoi audio ve trinh duyet (ban ~2 gio lam
    // cua so trang) - nut WAV/MP3 tai thang tu file qua downloadHistoryAudio.
    if (!(entry.audioBlob || entry.serverId)) {
      entryBlob(entry)
        .then((blob) => {
          if (currentEntryIdRef.current === entry.id) {
            setCurrentAudioBlob(blob);
          }
        })
        .catch((e) => console.warn("[History]", e));
    }
    if (!entry.subtitles && entry.subtitlesUrl) {
      entrySubtitles(entry).then((srt) => {
        if (currentEntryIdRef.current === entry.id) {
          setCurrentSubtitles(srt);
        }
      });
    }
  };

  const clearCurrentAudio = () => {
    sharedAudioRef.current?.pause();
    setIsAudioPlaying(false);
    currentEntryIdRef.current = null;
    setCurrentSubtitles(null);
    setCurrentAudioEntry(null);
    setCurrentAudioBlobUrl(null);
    setCurrentAudioBlob(null);
    setTtfaMs(null);
  };

  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const restartConnection = useCallback(() => {
    setIsInitialLoading(true);
    setConnectionAttempt((attempt) => attempt + 1);
  }, []);

  useEffect(() => {
    setConnectionStatus("connecting");
    setLoadingState("starting");
    setLoadingMessage("Connecting to the voice service...");
    setErrorMessage(null);

    return startStartupPolling({
      check: async (signal) => {
        const health = await checkServerHealth(settings.apiUrl, signal);
        if (signal.aborted) {
          return "failed";
        }
        setErrorMessage(null);
        setEngineName(health.engine || "VieNeu-TTS v3 Turbo");
        setConnectionStatus("connecting");

        if (health.status === "loading") {
          setLoadingState("starting");
          setLoadingMessage(health.message || "Loading voice model...");
          return "loading";
        }
        if (health.status === "error") {
          const message = health.message || "Could not load the voice model.";
          setConnectionStatus("disconnected");
          setLoadingState("failed");
          setLoadingMessage(message);
          setErrorMessage(message);
          return "failed";
        }
        if (health.status !== "ok") {
          throw new Error("Invalid server status.");
        }

        setLoadingMessage("Loading voice list...");
        const voiceList = await fetchVoices(settings.apiUrl, signal);
        if (signal.aborted) {
          return "failed";
        }
        audioEngineRef.current.setSampleRate(health.sample_rate || 48_000);
        setModelLanguages(health.languages ?? []);
        // Doi doan mau theo mo hinh, chi khi nguoi dung chua sua doan mau.
        const foreign = (health.languages ?? []).length > 0;
        setText((current) => {
          if (foreign && current === DEFAULT_TEXT) {
            return DEFAULT_TEXT_FOREIGN;
          }
          if (!foreign && current === DEFAULT_TEXT_FOREIGN) {
            return DEFAULT_TEXT;
          }
          return current;
        });
        setSupportsCloning(health.supports_cloning ?? true);
        setSupportsEmotions(health.supports_emotions ?? true);
        setSupportsSeed(health.supports_seed ?? true);
        setSettings((previous) => ({
          ...previous,
          sampleRate: health.sample_rate || 48_000,
        }));
        setVoices(voiceList);
        setSelectedVoiceId((previous) =>
          voiceList.some((voice) => voice.id === previous)
            ? previous
            : (voiceList[0]?.id ?? "")
        );
        setConnectionStatus("connected");
        setLoadingState("ready");
        setErrorMessage(null);
        setIsInitialLoading(false);
        loadHistory(settings.apiUrl);
        return "ready";
      },
      onConnectionError: (error, willRetry) => {
        console.warn("[Frontend] Connection failed:", error);
        setConnectionStatus(willRetry ? "connecting" : "disconnected");
        setLoadingState(willRetry ? "starting" : "failed");
        const message = willRetry
          ? "The voice service is not responding yet. The app is reconnecting automatically..."
          : `Could not connect to ${settings.apiUrl}. Click Retry or check the address in Settings.`;
        setLoadingMessage(message);
        setErrorMessage(willRetry ? null : message);
      },
    });
  }, [settings.apiUrl, connectionAttempt]);

  // Custom voice has been saved on the server (see VoiceCloningTab) - just reload
  // the list and select it; no fake localStorage bookmark anymore.
  const handleAddCustomVoice = async (voiceId: string) => {
    const voiceList = await fetchVoices(settings.apiUrl);
    if (!voiceList.some((voice) => voice.id === voiceId)) {
      throw new Error(
        "The newly saved voice was not found on the server. Please reload the voice list."
      );
    }
    setVoices(voiceList);
    setSelectedVoiceId(voiceId);
  };

  // Delete a saved cloned voice and reload the list. If the selected voice is deleted,
  // switch to the first voice in the list; other tabs receive the updated list via `voices`.
  const handleDeleteCustomVoice = async (voiceId: string) => {
    await deleteCustomVoice(settings.apiUrl, voiceId);
    const voiceList = await fetchVoices(settings.apiUrl);
    setVoices(voiceList);
    setSelectedVoiceId((previous) =>
      voiceList.some((voice) => voice.id === previous)
        ? previous
        : (voiceList[0]?.id ?? "")
    );
  };

  // Batch/standard generation has no real progress from the server (a single request
  // waits until done), so estimate based on text length + typical RTF of the
  // current engine so the progress bar is meaningful instead of just a spinner.
  const uocLuongMsXuLy = (soKyTu: number): number => {
    // Speed is applied after inference; it doesn't make the model generate audio faster.
    const giayNoi = soKyTu / 15;
    const ten = engineName.toLowerCase();
    let rtf = 0.5; // CPU int8, conservative default
    if (ten.includes("gpu")) {
      rtf = 0.15;
    } else if (ten.includes("fp32")) {
      rtf = 0.6;
    }
    return Math.max(300, giayNoi * rtf * 1000);
  };

  // Handle full audio generation (play after generation completes, no stuttering)
  const handleGenerateOrPlay = async () => {
    if (!text.trim()) {
      setErrorMessage(
        "Please enter text to convert to speech!"
      );
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setPlaybackStatus("generating");
    setTtfaMs(null);
    setErrorMessage(null);

    const startTime = performance.now();
    const estimatedMs = uocLuongMsXuLy(text.trim().length);
    setGenProgress({ elapsedMs: 0, percent: 0 });
    if (genTimerRef.current) {
      clearInterval(genTimerRef.current);
    }
    genTimerRef.current = setInterval(() => {
      const elapsedMs = performance.now() - startTime;
      setGenProgress({
        elapsedMs,
        percent: Math.min(95, 100 * (1 - Math.exp(-elapsedMs / estimatedMs))),
      });
    }, 100);

    try {
      const selectedVoice = voices.find((v) => v.id === selectedVoiceId);
      // The server saves to history on disk after generation; the browser only receives
      // the history item and plays/downloads via URL, without holding the full audio blob
      // (a ~2-hour file can blank the app window - see generateToHistory).
      const res = await generateToHistory(
        settings.apiUrl,
        {
          text: text.trim(),
          voiceId: selectedVoiceId,
          voiceName: selectedVoice ? selectedVoice.name : selectedVoiceId,
          speed,
          temperature,
          silenceDuration: settings.silenceDuration ?? 0.2,
          subtitles: withSubtitles,
          seed: stableVoice && supportsSeed ? STABLE_SEED : undefined,
        },
        controller.signal
      );

      if (
        controller.signal.aborted ||
        abortControllerRef.current !== controller
      ) {
        return;
      }
      const elapsedMs = Math.round(performance.now() - startTime);
      setTtfaMs(elapsedMs);

      // Item without audioBlob: handleAddHistory no longer uploads audio back.
      handleAddHistory({
        ...fromServer(settings.apiUrl, res.entry),
        subtitles: res.srt ?? undefined,
      });

      setPlaybackStatus("idle");
    } catch (err: any) {
      if (abortControllerRef.current !== controller) {
        return;
      }
      if (err.name === "AbortError") {
        setPlaybackStatus("idle");
      } else {
        console.error("[TTS Error]", err);
        setPlaybackStatus("error");
        setErrorMessage(err.message || "Error during speech generation");
      }
    } finally {
      // A cancelled request must not clear the progress of a newer request.
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
        if (genTimerRef.current) {
          clearInterval(genTimerRef.current);
          genTimerRef.current = null;
        }
        setGenProgress(null);
      }
    }
  };

  // Stop generation or audio playback
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (genTimerRef.current) {
      clearInterval(genTimerRef.current);
      genTimerRef.current = null;
    }
    setGenProgress(null);
    audioEngineRef.current.stop();
    setPlaybackStatus("idle");
  };

  // Ten goc chung cho WAV / MP3 / SRT cua cung mot ban audio: phan mem dung video
  // tu ghep phu de khi file .srt trung ten voi file audio.
  const currentFileBase = () => {
    if (currentAudioEntry) {
      return entryFileBase(currentAudioEntry);
    }
    const selectedVoice = voices.find((v) => v.id === selectedVoiceId);
    return `cit_tts_${selectedVoice ? selectedVoice.name : "voice"}_${Date.now()}`;
  };

  // Item saved on disk: download directly from file, not through the browser.
  const downloadFromDisk = (fmt: "wav" | "mp3") => {
    const id = currentAudioEntry?.serverId;
    if (!id) {
      return false;
    }
    downloadHistoryAudio(
      settings.apiUrl,
      id,
      `${currentFileBase()}.${fmt}`,
      fmt
    ).catch((e) => setErrorMessage(e instanceof Error ? e.message : String(e)));
    return true;
  };

  // Download WAV
  const handleDownloadWav = () => {
    if (currentAudioBlob) {
      downloadBlobAsFile(currentAudioBlob, `${currentFileBase()}.wav`);
      return;
    }
    downloadFromDisk("wav");
  };

  const handleDownloadSrt = () => {
    if (currentSubtitles) {
      downloadBlobAsFile(srtBlob(currentSubtitles), `${currentFileBase()}.srt`);
    }
  };

  // Download MP3
  const handleDownloadMp3 = async () => {
    if (!currentAudioBlob) {
      downloadFromDisk("mp3");
      return;
    }
    try {
      const mp3Blob = await convertToMp3(settings.apiUrl, currentAudioBlob);
      downloadBlobAsFile(mp3Blob, `${currentFileBase()}.mp3`);
    } catch {
      handleDownloadWav();
    }
  };

  // Transfer content from the Read File tab
  const handleSendToTtsFromPdf = (extractedText: string) => {
    setText(extractedText);
    setActiveTab("tts");
  };

  const handleSendToConversationFromPdf = () => {
    setActiveTab("conversation");
  };

  // Name of the selected voice in the status bar
  useEffect(() => {
    if (connectionStatus !== "connected" || appUpdate) {
      return;
    }
    const controller = new AbortController();
    checkAppUpdate(settings.apiUrl, false, controller.signal)
      .then((info) => {
        setAppUpdate(info);
        // Auto-check on app launch: show the dialog only once per version.
        if (info.newer && !info.skipped) {
          setUpdateDialogOpen(true);
        }
      })
      .catch(() => undefined); // mat mang / backend cu: im lang
    return () => controller.abort();
  }, [connectionStatus, settings.apiUrl, appUpdate]);

  const selectedVoiceName =
    voices.find((v) => v.id === selectedVoiceId)?.name ||
    selectedVoiceId ||
    "Default";

  const isGenerating =
    playbackStatus === "generating" || playbackStatus === "streaming";

  return (
    <ModelCapsProvider
      languages={modelLanguages}
      supportsCloning={supportsCloning}
      supportsEmotions={supportsEmotions}
    >
      <div
        className="flex h-screen select-none flex-col overflow-hidden bg-background font-sans text-foreground"
        onPlayCapture={(event) => {
          if (event.target instanceof HTMLAudioElement) {
            pauseOtherAudio(
              event.target,
              event.currentTarget.querySelectorAll("audio")
            );
          }
        }}
      >
        <TermsGate apiUrl={settings.apiUrl} />

        {/* Header */}
        <Header
          engineName={engineName}
          isDark={isDark}
          onOpenSettings={() => {
            setSettingsSection("models");
            setIsSettingsOpen(true);
          }}
          onRefresh={restartConnection}
          onToggleDark={() => {
            setIsDark(!isDark);
            persistUiPrefs({ theme: isDark ? "light" : "dark" });
          }}
          sampleRate={settings.sampleRate}
          status={connectionStatus}
        />

        {/* General error notice */}
        {errorMessage && (
          <div className="slide-in-from-top-2 mx-6 mt-3 flex animate-in items-center justify-between rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-rose-600 text-xs shadow-sm dark:text-rose-400">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              className="rounded-lg p-1 transition-colors hover:bg-rose-500/20"
              onClick={() => setErrorMessage(null)}
              type="button"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {!updateDismissed && (
          <UpdateBanner
            info={appUpdate}
            onDismiss={() => setUpdateDismissed(true)}
          />
        )}

        <div className="studio-body">
          <nav aria-label="Features" className="studio-sidebar">
            <div className="studio-nav-groups">
              {[
                {
                  title: "Create",
                  items: [
                    { tab: "tts", label: "Text to Speech", icon: Mic },
                    { tab: "conversation", label: "Conversation", icon: Users },
                    {
                      tab: "pdf",
                      label: "Read File, Dubbing",
                      icon: FileText,
                    },
                    { tab: "batch", label: "Batch", icon: ListOrdered },
                    { tab: "clone", label: "Voice Cloning", icon: Sparkles },
                  ],
                },
                {
                  title: "Library",
                  items: [
                    {
                      tab: "dictionary",
                      label: "Pronunciation Dictionary",
                      icon: BookOpen,
                    },
                    { tab: "history", label: "History", icon: History },
                  ],
                },
                {
                  title: "System",
                  items: [
                    { tab: "settings", label: "Settings", icon: Settings },
                    {
                      tab: "about",
                      label: "About & Credits",
                      icon: Info,
                    },
                  ],
                },
              ].map((group) => (
                <div className="studio-nav-group" key={group.title}>
                  <h2>{group.title}</h2>
                  {group.items.map((item) => (
                    <button
                      aria-current={activeTab === item.tab ? "page" : undefined}
                      className="studio-nav-item"
                      key={item.tab}
                      onClick={() => {
                        if (item.tab === "dictionary") {
                          setIsDictionaryOpen(true);
                        } else if (item.tab === "settings") {
                          setSettingsSection(undefined);
                          setIsSettingsOpen(true);
                        } else {
                          setActiveTab(item.tab as ActiveTab);
                        }
                      }}
                      type="button"
                    >
                      <item.icon aria-hidden="true" size={16} />
                      <span>{item.label}</span>
                      {item.tab === "history" && (
                        <span className="studio-nav-count">
                          {history.length}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              ))}
            </div>
            <SidebarFooter
              apiUrl={settings.apiUrl}
              info={appUpdate}
              onChecked={handleUpdateChecked}
            />
          </nav>
          <div
            className={`studio-workspace ${activeTab === "tts" ? "studio-workspace-tts" : ""}`}
          >
            {/* Workspace for each tab */}
            <main className="studio-main">
              {/* Single-voice text-to-speech tab */}
              {activeTab === "tts" && (
                <div className="studio-tts-layout">
                  {/* Middle column: text and emotion tools */}
                  <div className="studio-editor-area">
                    <TextEditor
                      disabled={isGenerating}
                      onChange={setText}
                      onOpenDictionary={() => setIsDictionaryOpen(true)}
                      text={text}
                    />
                  </div>

                  {/* Right panel: voice and generation options */}
                  <div className="studio-properties">
                    <LanguageSelect
                      className="rounded-xl border border-border/80 bg-card px-3 py-2 shadow-sm"
                      disabled={isGenerating}
                    />
                    <div className="min-h-0 flex-1">
                      <VoiceSelector
                        apiUrl={settings.apiUrl}
                        disabled={isGenerating}
                        onChangeSpeed={(s) => {
                          setSpeed(s);
                          audioEngineRef.current.setPlaybackRate(s);
                        }}
                        onCloneVoice={() => setActiveTab("clone")}
                        onDeleteVoice={handleDeleteCustomVoice}
                        onSelectVoice={setSelectedVoiceId}
                        selectedVoiceId={selectedVoiceId}
                        speed={speed}
                        voices={voices}
                      />
                    </div>
                    <details className="studio-generation-options">
                      <summary className="cursor-pointer py-2 font-semibold text-sm">
                        Advanced options
                      </summary>
                      <AdvancedGeneration
                        disabled={isGenerating}
                        onChangeTemperature={setTemperature}
                        temperature={temperature}
                      />
                      <label className="studio-switch-row">
                        <span className="flex items-center gap-1">
                          Include .srt subtitles
                          <InfoTip label="Include subtitles">
                            <strong>Generate with .srt file</strong>
                            <span>
                              Sentence-level timestamps for video editing.
                            </span>
                            <span>Download via Export → SRT.</span>
                          </InfoTip>
                        </span>
                        <input
                          checked={withSubtitles}
                          className="studio-switch"
                          disabled={isGenerating}
                          onChange={(event) =>
                            setWithSubtitles(event.target.checked)
                          }
                          type="checkbox"
                        />
                      </label>
                      {supportsSeed && (
                        <label className="studio-switch-row">
                          <span className="flex items-center gap-1">
                            Stable (keep voice consistent)
                            <InfoTip label="Stable">
                              <strong>Enabled: regenerate the exact same result</strong>
                              <span>
                                Including any mispronunciations — if that happens, turn it off and regenerate.
                              </span>
                              <span>
                                Disabled (default): intonation varies slightly each time.
                              </span>
                            </InfoTip>
                          </span>
                          <input
                            checked={stableVoice}
                            className="studio-switch"
                            disabled={isGenerating}
                            onChange={(event) =>
                              setStableVoice(event.target.checked)
                            }
                            type="checkbox"
                          />
                        </label>
                      )}
                    </details>
                    <div className="studio-generate-area">
                      <button
                        className="studio-primary studio-generate"
                        data-busy={isGenerating || undefined}
                        disabled={
                          connectionStatus !== "connected" || isGenerating
                        }
                        onClick={handleGenerateOrPlay}
                        type="button"
                      >
                        <Sparkles size={16} />
                        {isGenerating ? "Generating…" : "Generate speech"}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Voice cloning tab */}
              <div hidden={activeTab !== "clone"}>
                <VoiceCloningTab
                  apiUrl={settings.apiUrl}
                  isActive={activeTab === "clone"}
                  onAddCustomVoice={handleAddCustomVoice}
                  onAddHistory={handleAddHistory}
                />
              </div>

              {/* Multi-speaker conversation tab */}
              <div hidden={activeTab !== "conversation"}>
                <ConversationTab
                  active={activeTab === "conversation"}
                  apiUrl={settings.apiUrl}
                  onAddHistory={handleAddHistory}
                  restoredScript={restoredConversation}
                  voices={voices}
                />
              </div>

              {/* Tab 4: PDF Reader + long tieng SRT. Giu trong DOM (chi an): go ra khi doi
            tab se huy luot long tieng dang chay. */}
              <div hidden={activeTab !== "pdf"}>
                <PdfReaderTab
                  apiUrl={settings.apiUrl}
                  defaultVoiceId={selectedVoiceId}
                  onAddHistory={handleAddHistory}
                  onSendToConversation={handleSendToConversationFromPdf}
                  onSendToTts={handleSendToTtsFromPdf}
                  voices={voices}
                />
              </div>

              {/* Tao hang loat: LUON giu trong DOM (chi an) - go khoi DOM khi doi tab se
            huy tien trinh dang tao va mat cac file da tao. */}
              <div hidden={activeTab !== "batch"}>
                <BatchTab
                  apiUrl={settings.apiUrl}
                  defaultVoiceId={selectedVoiceId}
                  voices={voices}
                />
              </div>

              {/* History tab */}
              {activeTab === "history" && (
                <div className="space-y-4">
                  <HistoryStorageNote
                    apiUrl={settings.apiUrl}
                    refreshKey={history.length}
                  />
                  <HistoryList
                    items={history}
                    onClearAll={async () => {
                      // biome-ignore lint/suspicious/noAlert: confirm before deleting all saved files
                      const confirmed = window.confirm(
                        "Delete all generated items and saved audio files?"
                      );
                      if (!confirmed) {
                        return;
                      }
                      const ids = new Set(history.map((item) => item.id));
                      try {
                        await clearHistory(settings.apiUrl);
                        if (
                          currentEntryIdRef.current &&
                          ids.has(currentEntryIdRef.current)
                        ) {
                          clearCurrentAudio();
                        }
                        setHistory((prev) =>
                          prev.filter((item) => !ids.has(item.id))
                        );
                      } catch {
                        setErrorMessage(
                          "Could not delete generated items. Please try again."
                        );
                      }
                    }}
                    onDeleteItem={async (id) => {
                      const item = history.find((h) => h.id === id);
                      try {
                        if (item?.serverId) {
                          await deleteHistory(settings.apiUrl, item.serverId);
                        }
                        if (currentEntryIdRef.current === id) {
                          clearCurrentAudio();
                        }
                        setHistory((prev) => prev.filter((h) => h.id !== id));
                      } catch {
                        setErrorMessage(
                          "Could not delete this item. Please try again."
                        );
                      }
                    }}
                    onDownloadItem={(item) => {
                      if (item.serverId && !item.audioBlob) {
                        downloadHistoryAudio(
                          settings.apiUrl,
                          item.serverId,
                          `${entryFileBase(item)}.${item.format ?? "wav"}`,
                          item.format ?? "wav"
                        ).catch((e) =>
                          setErrorMessage(
                            e instanceof Error ? e.message : String(e)
                          )
                        );
                        return;
                      }
                      entryBlob(item)
                        .then((blob) =>
                          downloadBlobAsFile(
                            blob,
                            `${entryFileBase(item)}.${item.format ?? "wav"}`
                          )
                        )
                        .catch((e) => console.warn("[History]", e));
                    }}
                    onPlayItem={handlePlayHistory}
                    onRestoreText={(item) => {
                      if (item.mode === "conversation") {
                        setRestoredConversation({
                          key: Date.now(),
                          script: item.text,
                        });
                        setActiveTab("conversation");
                        return;
                      }
                      setText(item.text);
                      setActiveTab("tts");
                    }}
                    onRevealItem={(item) => {
                      if (!item.serverId) {
                        return;
                      }
                      revealHistory(settings.apiUrl, item.serverId).catch((e) =>
                        // biome-ignore lint/suspicious/noAlert: bao loi ngan gon, hiem khi xay ra
                        window.alert(e instanceof Error ? e.message : String(e))
                      );
                    }}
                    playingId={
                      isAudioPlaying ? (currentAudioEntry?.id ?? null) : null
                    }
                  />
                </div>
              )}

              {/* About and credits tab */}
              {activeTab === "about" && (
                <AboutTab
                  apiUrl={settings.apiUrl}
                  engineName={engineName}
                  onUpdateChecked={handleUpdateChecked}
                  sampleRate={settings.sampleRate}
                  update={appUpdate}
                />
              )}
            </main>

            {/* Shared player, kept in the DOM when switching tabs */}
            {/* Chi hien o tab Tao giong (soan + tao) va Lich su (nghe/tai lai). Cac tab
          nhan ban, hoi thoai, hang loat co nut tao + trinh phat rieng; nut "Tao giong
          noi" o day doc van ban tab Tao giong nen de bam nham. Van hien khi dang tao
          (con nut "Dung tao") hoac dang phat (con nut tam dung). */}
            <footer
              className="studio-player"
              hidden={
                !(
                  activeTab === "tts" ||
                  activeTab === "history" ||
                  playbackStatus === "generating" ||
                  isAudioPlaying
                )
              }
            >
              <AudioPlayerBar
                audioBlob={currentAudioBlob}
                audioBlobUrl={currentAudioBlobUrl}
                audioRef={sharedAudioRef}
                canDownload={Boolean(currentAudioEntry?.serverId)}
                disabled={connectionStatus !== "connected"}
                getVisualizerData={() =>
                  audioEngineRef.current.getFrequencyData()
                }
                isPlaying={isAudioPlaying}
                onChangeVolume={(v) => {
                  setVolume(v);
                  audioEngineRef.current.setVolume(v);
                }}
                onDownload={handleDownloadWav}
                onDownloadMp3={handleDownloadMp3}
                onDownloadSrt={currentSubtitles ? handleDownloadSrt : undefined}
                onGenerateOrPlay={undefined}
                onPlayingChange={setIsAudioPlaying}
                onStop={handleStop}
                playbackSpeed={speed}
                progress={genProgress}
                selectedVoiceName={
                  currentAudioEntry?.voiceName ?? selectedVoiceName
                }
                stableEnabled={stableVoice && supportsSeed}
                status={playbackStatus}
                subtitlesEnabled={withSubtitles}
                ttfaMs={ttfaMs}
                volume={volume}
              />
            </footer>
          </div>
        </div>

        <PronunciationDictionaryModal
          apiUrl={settings.apiUrl}
          isOpen={isDictionaryOpen}
          onClose={() => setIsDictionaryOpen(false)}
        />

        {/* "New version available" dialog. */}
        {updateDialogOpen && appUpdate?.newer && (
          <UpdateDialog
            apiUrl={settings.apiUrl}
            info={appUpdate}
            onClose={closeUpdateDialog}
          />
        )}

        {/* "Saved" notification after each file save (desktop app). */}
        <SavedFileToast />

        {/* Settings dialog */}
        <SettingsModal
          initialSection={settingsSection}
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          onModelSwitched={(model) => {
            setEngineName(model);
            appliedDefaultVoice.current = "";
            restartConnection();
          }}
          onSaveSettings={(newSettings) => {
            setSettings(newSettings);
            localStorage.setItem(
              "hupesi_tts_settings",
              JSON.stringify(newSettings)
            );
            persistUiPrefs({
              autoPlayOnGenerate: newSettings.autoPlayOnGenerate,
              defaultVoiceId: newSettings.defaultVoiceId,
              playbackSpeed: newSettings.playbackSpeed,
              silenceDuration: newSettings.silenceDuration,
            });
          }}
          onUpdateChecked={handleUpdateChecked}
          settings={settings}
          voices={voices}
        />

        {/* Startup loading screen */}
        {isInitialLoading && !isSettingsOpen && (
          <LoadingScreen
            isDark={isDark}
            message={loadingMessage}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onRetry={restartConnection}
            status={loadingState}
          />
        )}
      </div>
    </ModelCapsProvider>
  );
};
