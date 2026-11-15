import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileUp,
  Mic,
  Save,
  ShieldCheck,
  Sparkles,
  Square,
  Volume2,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  addCustomVoice,
  type ConsentInfo,
  fetchConsentInfo,
  generateToHistory,
} from "../api";
import { convertAudioToWav, ReferenceTooLongError } from "../audio-engine";
import { downloadHistoryAudio, fromServer } from "../history";
import { useModelCaps } from "../model-caps";
import { REF_MAX_SECONDS } from "../ref-audio";
import type { HistoryEntry } from "../types";
import { useCaretInsert } from "../use-caret-insert";
import { EmotionBar } from "./EmotionBar";

const errorText = (err: unknown, fallback: string) =>
  err instanceof Error && err.message ? err.message : fallback;

interface VoiceCloningTabProps {
  apiUrl: string;
  isActive: boolean;
  onAddCustomVoice: (voiceId: string) => Promise<void>;
  onAddHistory: (entry: HistoryEntry) => void;
}

// Microphone and audio file inputs share the same usage consent and do not require ASR.
// Both save the voice and consent locally and do not verify user identity.

// Files are decoded in the browser and sent to the server as short WAV audio, so
// any format WebView2 can decode is supported; only common extensions are exposed.
const SOURCE_EXTENSIONS = [".mp3", ".wav", ".m4a"];
const SOURCE_ACCEPT = `${SOURCE_EXTENSIONS.join(",")},audio/mpeg,audio/wav,audio/mp4`;
const SOURCE_FORMATS = "MP3, WAV, or M4A";
export const VoiceCloningTab: React.FC<VoiceCloningTabProps> = ({
  apiUrl,
  isActive,
  onAddCustomVoice,
  onAddHistory,
}) => {
  const { supportsCloning } = useModelCaps();
  // Step 1: confirm usage rights.
  const [agreed, setAgreed] = useState(false);
  const [inputSource, setInputSource] = useState<"recording" | "file">(
    "recording"
  );
  const [isPreparing, setIsPreparing] = useState(false);
  const [sourceFileName, setSourceFileName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [consent, setConsent] = useState<ConsentInfo | null>(null);
  const [consentError, setConsentError] = useState<string | null>(null);

  // Step 2: record audio
  const [audioFile, setAudioFile] = useState<Blob | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Step 3: review & save
  const [voiceName, setVoiceName] = useState("My voice");
  const [denoise, setDenoise] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedVoice, setSavedVoice] = useState<string | null>(null);

  // Preview the newly created voice
  const [targetText, setTargetText] = useState(
    "Hello everyone! This is an AI-cloned voice."
  );
  const [speed, setSpeed] = useState(1.0);
  const [isGenerating, setIsGenerating] = useState(false);
  const caret = useCaretInsert(targetText, setTargetText, isGenerating);
  const [outputEntryId, setOutputEntryId] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refreshConsent = useCallback(async () => {
    try {
      setConsent(await fetchConsentInfo(apiUrl));
      setConsentError(null);
    } catch (err) {
      setConsentError(
        errorText(
          err,
          "Unable to load the consent terms. Check the app connection and reopen this tab."
        )
      );
    }
  }, [apiUrl]);

  useEffect(() => {
    if (isActive) {
      refreshConsent();
    }
  }, [isActive, refreshConsent]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (recordTimerRef.current) {
      clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
  }, []);

  // Stop the microphone when leaving the tab (keep the draft).
  useEffect(() => {
    if (!isActive) {
      stopRecording();
    }
  }, [isActive, stopRecording]);

  useEffect(() => {
    return () => {
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
    };
  }, [audioPreviewUrl]);

  const handleRecorded = async (raw: Blob) => {
    setIsPreparing(true);
    setErrorMessage(null);
    try {
      const { wavBlob, duration } = await convertAudioToWav(
        raw,
        REF_MAX_SECONDS
      );
      setAudioFile(wavBlob);
      setAudioPreviewUrl(URL.createObjectURL(wavBlob));
      setAudioDuration(duration);
      setSavedVoice(null);
      setOutputUrl(null);
      setOutputEntryId(null);
      setSaveError(null);
    } catch (err) {
      setAudioFile(null);
      setAudioPreviewUrl(null);
      setAudioDuration(null);
      setErrorMessage(
        err instanceof ReferenceTooLongError
          ? "The file is too long to process. Trim a clear 5–15 second voice clip and try again."
          : `Unable to read the audio. Please choose a valid ${SOURCE_FORMATS} file or record again.`
      );
    } finally {
      setIsPreparing(false);
    }
  };

  const changeSource = (source: "recording" | "file") => {
    if (source === inputSource) {
      return;
    }
    setInputSource(source);
    setAudioFile(null);
    setAudioPreviewUrl(null);
    setAudioDuration(null);
    setSourceFileName("");
    setSavedVoice(null);
    setSaveError(null);
    setErrorMessage(null);
    setOutputUrl(null);
    setOutputEntryId(null);
  };

  const handleFileSelected = async (file?: File) => {
    if (!file) {
      return;
    }
    setSavedVoice(null);
    setAudioFile(null);
    setAudioPreviewUrl(null);
    setAudioDuration(null);
    setSaveError(null);
    setSourceFileName(file.name);
    const lowerName = file.name.toLowerCase();
    if (!SOURCE_EXTENSIONS.some((ext) => lowerName.endsWith(ext))) {
      setErrorMessage(
        `Please choose a ${SOURCE_FORMATS} file. The sample should contain one clear speaker with no background music.`
      );
      return;
    }
    await handleRecorded(file);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      recorder.onstop = async () => {
        for (const track of stream.getTracks()) {
          track.stop();
        }
        await handleRecorded(
          new Blob(audioChunksRef.current, {
            type: recorder.mimeType || "audio/webm",
          })
        );
      };
      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setIsRecording(true);
      setRecordSeconds(0);
      setErrorMessage(null);
      recordTimerRef.current = setInterval(
        () => setRecordSeconds((s) => s + 1),
        1000
      );
    } catch (err) {
      setErrorMessage(
        `Unable to access the microphone: ${errorText(err, String(err))}. Allow the app to use the microphone in your operating system privacy settings.`
      );
    }
  };

  const handleSave = async () => {
    const name = voiceName.trim();
    if (!(name && audioFile && agreed && consent?.terms_version)) {
      return;
    }
    setIsSaving(true);
    setSaveError(null);
    try {
      const saved = await addCustomVoice(apiUrl, {
        name,
        audioFile,
        denoise,
        inputSource,
        rightsConfirmed: agreed,
        rightsVersion: consent.terms_version,
      });
      await onAddCustomVoice(saved.voice_id);
      setSavedVoice(saved.voice_id);
    } catch (err) {
      setSaveError(errorText(err, "Unable to save the voice."));
    } finally {
      setIsSaving(false);
    }
  };

  const handleGenerate = async () => {
    if (!(savedVoice && targetText.trim())) {
      return;
    }
    setIsGenerating(true);
    setErrorMessage(null);
    try {
      const res = await generateToHistory(apiUrl, {
        text: targetText.trim(),
        voiceId: savedVoice,
        voiceName: savedVoice,
        speed,
      });
      const entry = fromServer(apiUrl, res.entry);
      setOutputEntryId(res.entry.id);
      setOutputUrl(entry.audioBlobUrl);
      onAddHistory(entry);
    } catch (err) {
      setErrorMessage(errorText(err, "Error while previewing the voice."));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = async (fmt: "wav" | "mp3") => {
    if (!outputEntryId) {
      return;
    }
    try {
      await downloadHistoryAudio(
        apiUrl,
        outputEntryId,
        `cit_${savedVoice || "voice"}_${Date.now()}.${fmt}`,
        fmt
      );
    } catch (err) {
      setErrorMessage(errorText(err, "Error downloading the file."));
    }
  };

  const canRecord = agreed && !isSaving && !isPreparing;
  const stepBadge = (n: number) => (
    <span className="flex h-6 w-6 items-center justify-center rounded-full border border-brand-500/20 bg-brand-500/10 font-bold text-brand-600 text-xs dark:text-brand-400">
      {n}
    </span>
  );

  let saveLabel = "Save voice on this device";
  if (isSaving) {
    saveLabel = "Creating & saving voice…";
  } else if (savedVoice) {
    saveLabel = "Saved to voice library";
  }

  // Foreign-language models (Supertonic) only provide preset voices; the server also returns 400.
  if (!supportsCloning) {
    return (
      <div className="studio-tab-card mx-auto max-w-xl space-y-2 rounded-[14px] border border-border bg-card p-6 text-center shadow-sm">
        <Sparkles className="mx-auto h-6 w-6 text-brand-500" />
        <h3 className="font-bold text-foreground text-sm">
          The current model does not support voice cloning
        </h3>
        <p className="text-muted-foreground text-xs leading-relaxed">
          Foreign-language models only provide preset voices. To clone your own voice,
          open <strong>Settings → Models</strong> and switch to{" "}
          <strong>VieNeu-TTS v3 Turbo</strong>.
        </p>
      </div>
    );
  }

  return (
    <div
      className="studio-tab-grid studio-clone-layout fade-in-50 animate-in duration-300"
      data-agreed={agreed}
    >
      <div className="studio-span-5 clone-create">
        {/* Step 1: confirm voice ownership */}
        <div className="studio-tab-card clone-consent space-y-2.5 rounded-[14px] border border-border bg-card p-4 shadow-sm">
          <div className="studio-section-heading flex items-center gap-2 border-border/60 border-b pb-2.5">
            {stepBadge(1)}
            <h3 className="font-bold text-foreground text-sm">
              Choose voice source
            </h3>
          </div>
          <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-brand-500/20 bg-brand-500/5 px-3 py-2 text-brand-800 text-xs leading-relaxed dark:text-brand-200">
            <input
              checked={agreed}
              className="mt-0.5 h-4 w-4 shrink-0 accent-brand-600"
              disabled={isSaving || isPreparing || !consent?.rights_statement}
              onChange={(event) => {
                setAgreed(event.target.checked);
                if (!event.target.checked) {
                  stopRecording();
                }
              }}
              type="checkbox"
            />
            <span>{consent?.rights_statement}</span>
          </label>
          {consentError && (
            <p className="text-rose-600 text-xs" role="alert">
              {consentError}
            </p>
          )}
          <div className="clone-sources flex gap-2" inert={!agreed}>
            <button
              aria-pressed={inputSource === "recording"}
              className={`flex flex-1 items-center justify-center gap-2 rounded-lg border px-3 py-2 font-semibold text-xs ${inputSource === "recording" ? "border-brand-500 bg-brand-500/10 text-brand-700 dark:text-brand-300" : "border-border bg-card text-foreground"}`}
              disabled={isRecording || isSaving || isPreparing}
              onClick={() => changeSource("recording")}
              type="button"
            >
              <Mic aria-hidden="true" size={16} /> Record with microphone
            </button>
            <button
              aria-pressed={inputSource === "file"}
              className={`flex flex-1 items-center justify-center gap-2 rounded-lg border px-3 py-2 font-semibold text-xs ${inputSource === "file" ? "border-brand-500 bg-brand-500/10 text-brand-700 dark:text-brand-300" : "border-border bg-card text-foreground"}`}
              disabled={isRecording || isSaving || isPreparing}
              onClick={() => changeSource("file")}
              type="button"
            >
              <FileUp aria-hidden="true" size={16} /> Upload audio file
            </button>
          </div>
        </div>

        <div className="clone-locked space-y-3" inert={!agreed}>
          {/* Step 2: record or upload a sample */}
          <div className="studio-tab-card space-y-2.5 rounded-[14px] border border-border bg-card p-4 shadow-sm">
            <div className="studio-section-heading flex items-center gap-2 border-border/60 border-b pb-2.5">
              {stepBadge(2)}
              <h3 className="flex-1 font-bold text-foreground text-sm">
                {inputSource === "recording"
                  ? "Record a sample"
                  : "Choose a voice sample"}
              </h3>
              {inputSource === "recording" && (
                <span className="font-bold font-mono text-foreground text-sm">
                  {String(Math.floor(recordSeconds / 60)).padStart(2, "0")}:
                  {String(recordSeconds % 60).padStart(2, "0")}
                </span>
              )}
              {inputSource === "recording" &&
                (isRecording ? (
                  <button
                    className="flex animate-pulse items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 font-semibold text-white text-xs"
                    onClick={stopRecording}
                    type="button"
                  >
                    <Square className="h-3.5 w-3.5 fill-white" />
                    Stop
                  </button>
                ) : (
                  <button
                    className="flex items-center gap-1.5 rounded-lg bg-rose-500 px-3 py-1.5 font-semibold text-white text-xs disabled:cursor-not-allowed"
                    disabled={!canRecord}
                    onClick={startRecording}
                    type="button"
                  >
                    <Mic className="h-3.5 w-3.5" />
                    {audioFile ? "Record again" : "Record"}
                  </button>
                ))}
            </div>
            {inputSource === "file" ? (
              <div className="space-y-2">
                <input
                  accept={SOURCE_ACCEPT}
                  aria-label="Select an audio file as a voice sample"
                  className="hidden"
                  disabled={isSaving || isPreparing}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    handleFileSelected(file);
                  }}
                  ref={fileInputRef}
                  type="file"
                />
                <button
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-border border-dashed bg-secondary/30 p-4 font-semibold text-foreground text-sm hover:bg-secondary disabled:opacity-50"
                  disabled={isSaving || isPreparing}
                  onClick={() => fileInputRef.current?.click()}
                  type="button"
                >
                  <FileUp aria-hidden="true" size={18} />
                  {isPreparing
                    ? "Preparing sample…"
                    : `Choose a ${SOURCE_FORMATS} file from your device`}
                </button>
                {sourceFileName && (
                  <p className="break-all text-[12px] text-muted-foreground">
                    {sourceFileName}
                  </p>
                )}
                <p className="text-[12px] text-muted-foreground">
                  One clear speaker, no background music; keep it under 15 seconds.
                </p>
              </div>
            ) : (
              <>
                <p className="clone-sample select-text rounded-lg border border-border/60 bg-secondary/30 px-3 py-2 font-semibold text-foreground text-sm leading-relaxed">
                  “This is my voice. I agree to create an AI voice to tell
                  meaningful stories!”
                </p>
                <p className="text-[12px] text-muted-foreground">
                  Read naturally for 3–8 seconds in a quiet place.
                </p>
              </>
            )}
            {audioPreviewUrl && (
              <div className="flex items-center gap-2">
                <Volume2 className="h-3.5 w-3.5 shrink-0 text-brand-500" />
                {/* biome-ignore lint/a11y/useMediaCaption: user-selected reference recording with a preview control */}
                <audio className="h-8 flex-1" controls src={audioPreviewUrl} />
                {audioDuration !== null && (
                  <span className="font-mono text-[12px]">
                    {audioDuration.toFixed(1)}s
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Step 3: review & save */}
          <div
            className={`studio-tab-card space-y-2.5 rounded-[14px] border border-border bg-card p-4 shadow-sm ${audioFile ? "" : "opacity-50"}`}
          >
            <div className="studio-section-heading flex items-center gap-2 border-border/60 border-b pb-2.5">
              {stepBadge(3)}
              <h3 className="font-bold text-foreground text-sm">
                Create &amp; save voice
              </h3>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <input
                aria-label="Voice name"
                className="min-w-0 flex-1 rounded-lg border border-border/80 bg-background px-2.5 py-1.5 outline-none focus:border-brand-500"
                disabled={isSaving}
                onChange={(e) => setVoiceName(e.target.value)}
                placeholder="Voice name"
                type="text"
                value={voiceName}
              />
              <label className="flex shrink-0 cursor-pointer items-center gap-1.5">
                <input
                  checked={denoise}
                  disabled={isSaving}
                  onChange={(e) => setDenoise(e.target.checked)}
                  type="checkbox"
                />
                Noise reduction
              </label>
            </div>
            <p className="text-[12px] text-muted-foreground">
              The voice and consent are stored only on your device.
            </p>
            <button
              className="studio-primary flex w-full items-center justify-center gap-2 rounded-lg bg-brand-600 px-4 py-2 font-semibold text-white text-xs disabled:cursor-not-allowed disabled:opacity-50"
              disabled={
                !(
                  audioFile &&
                  agreed &&
                  voiceName.trim() &&
                  consent?.terms_version
                ) ||
                isSaving ||
                isPreparing ||
                isRecording ||
                savedVoice !== null
              }
              onClick={handleSave}
              type="button"
            >
              {savedVoice ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <ShieldCheck className="h-4 w-4" />
              )}
              {saveLabel}
            </button>
            {savedVoice && (
              <p className="text-[12px] text-muted-foreground" role="status">
                Saved “{savedVoice}”. Select this voice in TTS Studio or Conversation,
                or call the API with voice_id = “{savedVoice}”.
              </p>
            )}
            {saveError && (
              <p className="text-rose-600 text-xs" role="alert">
                {saveError}
              </p>
            )}
          </div>
        </div>

        {/* Right column: preview the newly created voice */}
      </div>
      <div
        className="studio-tab-card studio-span-7 clone-preview flex flex-col space-y-2.5 rounded-[14px] border border-border bg-card p-4 shadow-sm"
        inert={!agreed}
      >
        <div className="studio-section-heading flex items-center gap-2 border-border/60 border-b pb-2.5">
          <Save className="h-4 w-4 text-brand-500" />
          <h3 className="flex-1 font-bold text-foreground text-sm">
            Preview the newly created voice
          </h3>
          <span className="text-[12px] text-muted-foreground">
            Speed {speed.toFixed(2)}x
          </span>
          <input
            aria-label="Speed"
            className="h-1.5 w-28 cursor-pointer appearance-none rounded-lg bg-secondary accent-brand-500"
            max="1.5"
            min="0.75"
            onChange={(e) => setSpeed(Number.parseFloat(e.target.value))}
            step="0.05"
            type="range"
            value={speed}
          />
        </div>
        <EmotionBar
          disabled={isGenerating || !savedVoice}
          onInsertTag={caret.insertTag}
        />
        <textarea
          className="w-full resize-none rounded-xl border border-border/80 bg-secondary/20 p-3 text-[14px] text-foreground leading-relaxed outline-none focus:border-brand-500"
          disabled={isGenerating || !savedVoice}
          onChange={(e) => setTargetText(e.target.value)}
          onFocus={caret.onFocus}
          ref={caret.textareaRef}
          rows={3}
          value={targetText}
        />
        <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-amber-700 text-xs dark:text-amber-300">
          <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
          <strong>
            When publishing publicly, you must clearly state that the voice was generated by AI.
          </strong>
        </div>
        <button
          className="studio-primary flex items-center justify-center gap-2.5 rounded-xl bg-gradient-to-r from-brand-500 to-teal-600 px-6 py-2.5 font-bold text-sm text-white disabled:cursor-not-allowed disabled:opacity-50"
          disabled={!(savedVoice && targetText.trim()) || isGenerating}
          onClick={handleGenerate}
          type="button"
        >
          <Sparkles className="h-4 w-4" />
          {isGenerating ? "Reading..." : "Preview"}
        </button>
        {errorMessage && (
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-rose-600 text-xs dark:text-rose-400">
            <AlertTriangle className="h-4 w-4 flex-shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}
        {outputUrl && (
          <div className="flex items-center gap-2">
            {/* biome-ignore lint/a11y/useMediaCaption: synthesized audio; the text is in the field above */}
            <audio className="h-9 flex-1" controls src={outputUrl} />
            <button
              className="flex items-center gap-1 rounded-lg border border-border bg-secondary px-2.5 py-1.5 font-medium text-foreground text-xs"
              onClick={() => handleDownload("wav")}
              type="button"
            >
              <Download className="h-3.5 w-3.5 text-brand-500" />
              WAV
            </button>
            <button
              className="flex items-center gap-1 rounded-lg border border-border bg-secondary px-2.5 py-1.5 font-medium text-foreground text-xs"
              onClick={() => handleDownload("mp3")}
              type="button"
            >
              <Download className="h-3.5 w-3.5 text-cyan-500" />
              MP3
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
