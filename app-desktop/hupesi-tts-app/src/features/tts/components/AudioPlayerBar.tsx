import {
  Anchor,
  Captions,
  Download,
  Pause,
  Play,
  Square,
  Volume2,
  VolumeX,
  Zap,
} from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { toggleAudioPlayback } from "../playback";
import type { PlaybackStatus } from "../types";

interface AudioPlayerBarProps {
  audioBlob: Blob | null;
  audioBlobUrl: string | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  /** Items saved to disk can be downloaded even without a blob in the browser. */
  canDownload?: boolean;
  disabled?: boolean;
  getVisualizerData?: () => Uint8Array | null;
  isPlaying: boolean;
  onChangeVolume: (vol: number) => void;
  onDownload: () => void;
  onDownloadMp3?: () => void;
  /** Download the .srt subtitles for the current audio; undefined = this item has no subtitles. */
  onDownloadSrt?: () => void;
  /** Omit in tabs without a text editor: hides the "Text to Speech" button (the "Stop generating" button still appears while generating). */
  onGenerateOrPlay?: () => void;
  onPlayingChange: (playing: boolean) => void;
  onStop: () => void;
  /** "Keep stable": uses a fixed seed so the same text + voice produces identical output. Only
   * passed by TTS Studio; omit to hide the toggle. */
  onToggleStable?: (enabled: boolean) => void;
  onToggleSubtitles?: (enabled: boolean) => void;
  playbackSpeed?: number;
  /** Estimated progress while generating speech (no real progress is provided by the server
   * for batch-style requests). `null` when not generating. */
  progress?: { elapsedMs: number; percent: number } | null;
  selectedVoiceName: string;
  stableEnabled?: boolean;
  status: PlaybackStatus;
  subtitlesEnabled?: boolean;
  ttfaMs: number | null;
  volume: number;
}

/** Height of wave bar `i`: based on analyzer data while playing, a simulated wave when
 *  no data is available yet, and a static wave while idle. */
const waveBarHeight = (
  i: number,
  numBars: number,
  isLive: boolean,
  data: Uint8Array | null,
  height: number
): number => {
  if (!isLive) {
    return 4 + Math.abs(Math.sin(i * 0.9) * Math.cos(i * 0.37)) * 16;
  }
  if (data && data.length > 0) {
    const dataIdx = Math.floor((i / numBars) * data.length);
    return Math.max(3, (data[dataIdx] / 255) * height);
  }
  return 3 + Math.sin(Date.now() / 150 + i) * 6;
};

export const AudioPlayerBar: React.FC<AudioPlayerBarProps> = ({
  status,
  playbackSpeed = 1,
  ttfaMs,
  audioBlobUrl,
  audioBlob,
  canDownload,
  audioRef: audioElRef,
  isPlaying,
  onPlayingChange,
  selectedVoiceName,
  onGenerateOrPlay,
  onStop,
  onDownload,
  onDownloadMp3,
  onDownloadSrt,
  onToggleSubtitles,
  subtitlesEnabled,
  onToggleStable,
  stableEnabled,
  volume,
  onChangeVolume,
  getVisualizerData,
  disabled,
  progress,
}) => {
  const downloadable = Boolean(audioBlob) || Boolean(canDownload);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const exportRef = useRef<HTMLDetailsElement | null>(null);

  // The "Export file" menu is a <details> element: it only closes when its summary is clicked again. Also close it
  // when clicking outside, pressing Esc, or after selecting a format.
  useEffect(() => {
    const close = () => exportRef.current?.removeAttribute("open");
    const onPointerDown = (e: PointerEvent) => {
      const menu = exportRef.current;
      if (menu?.open && !menu.contains(e.target as Node)) {
        close();
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        close();
      }
    };
    const onMenuClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target.closest(".studio-export-menu button:not(:disabled)")) {
        close();
      }
    };
    const menu = exportRef.current;
    menu?.addEventListener("click", onMenuClick);
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      menu?.removeEventListener("click", onMenuClick);
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, []);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);

  // Sync volume with the audio element
  useEffect(() => {
    if (audioElRef.current) {
      audioElRef.current.volume = volume;
    }
  }, [volume]);

  // Handle audio element events
  const handleTimeUpdate = () => {
    if (audioElRef.current) {
      setCurrentTime(audioElRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioElRef.current) {
      setDuration(audioElRef.current.duration || 0);
    }
  };

  const togglePlayPause = () => {
    if (!audioElRef.current) {
      return;
    }
    void toggleAudioPlayback(audioElRef.current);
  };

  const handleSeek = (time: number) => {
    if (audioElRef.current) {
      audioElRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  // Both history and the bottom bar share this single audio player.
  useEffect(() => {
    const audio = audioElRef.current;
    setCurrentTime(0);
    setDuration(0);
    onPlayingChange(false);
    if (!(audio && audioBlobUrl)) {
      return;
    }
    audio.pause();
    audio.currentTime = 0;
    void toggleAudioPlayback(audio);
    return () => audio.pause();
  }, [audioBlobUrl, audioElRef, onPlayingChange]);

  // Canvas waveform rendering loop
  useEffect(() => {
    let animId = 0;
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const render = () => {
      const data = getVisualizerData ? getVisualizerData() : null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const isLive = isPlaying;
      const numBars = 44;
      const barWidth = canvas.width / numBars - 2;
      const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
      // Teal -> cyan, readable on both light and dark backgrounds.
      grad.addColorStop(0, "#14b8a6");
      grad.addColorStop(1, "#06b6d4");
      ctx.fillStyle = grad;
      // When idle, draw a faint static wave instead of a gray bar so the player doesn't look disabled.
      ctx.globalAlpha = isLive ? 1 : 0.35;

      for (let i = 0; i < numBars; i++) {
        const val = waveBarHeight(i, numBars, isLive, data, canvas.height);

        const x = i * (barWidth + 2);
        const y = (canvas.height - val) / 2;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, val, 2);
        ctx.fill();
      }

      if (isLive) {
        animId = requestAnimationFrame(render);
      }
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, getVisualizerData]);

  const isGenerating = status === "generating";
  const isBusy = isGenerating;
  const progressPercent = Math.min(
    95,
    Math.max(0, Math.round(progress?.percent ?? 0))
  );

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? "0" : ""}${s}`;
  };

  return (
    <div className="studio-audio-player">
      {/* Hidden audio element */}
      <audio
        className="hidden"
        onEmptied={() => onPlayingChange(false)}
        onEnded={() => onPlayingChange(false)}
        onError={() => onPlayingChange(false)}
        onLoadedMetadata={handleLoadedMetadata}
        onPause={() => onPlayingChange(false)}
        onPlay={() => onPlayingChange(true)}
        onTimeUpdate={handleTimeUpdate}
        ref={audioElRef}
        src={audioBlobUrl ?? undefined}
      />

      <button
        aria-label={isPlaying ? "Pause" : "Play"}
        className="studio-play-button"
        data-playing={isPlaying || undefined}
        disabled={!audioBlobUrl}
        onClick={togglePlayPause}
        type="button"
      >
        {isPlaying ? (
          <Pause size={18} />
        ) : (
          <Play className="fill-current" size={18} />
        )}
      </button>
      {/* Stop generation button and playback information. */}
      <div className="flex w-full items-center gap-3 md:w-auto">
        {(onGenerateOrPlay || isBusy) && (
          <button
            className={`flex flex-1 select-none items-center justify-center gap-2 rounded-lg px-4 py-2 font-bold text-xs shadow-xs transition-all active:scale-95 sm:text-sm md:flex-none ${
              isBusy
                ? "animate-pulse bg-rose-500 text-white shadow-rose-500/20 hover:bg-rose-600"
                : "bg-gradient-to-r from-brand-500 to-teal-600 text-white shadow-brand-500/25 hover:from-brand-600 hover:to-teal-700"
            } ${disabled && !isBusy ? "cursor-not-allowed opacity-50" : ""}`}
            disabled={disabled && !isBusy}
            onClick={isBusy ? onStop : onGenerateOrPlay}
            type="button"
          >
            {isBusy ? (
              <>
                <Square className="h-4 w-4 fill-white" />
                <span>Stop generating</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-white" />
                <span>Text to Speech</span>
              </>
            )}
          </button>
        )}

        {onToggleSubtitles && (
          <label
            className={`flex shrink-0 cursor-pointer select-none items-center gap-1.5 rounded-lg border px-2 py-1.5 font-semibold text-xs transition-colors ${
              subtitlesEnabled
                ? "border-violet-500/60 bg-violet-500/10 text-violet-700 dark:text-violet-300"
                : "border-border text-muted-foreground hover:text-foreground"
            } ${isBusy ? "cursor-not-allowed opacity-50" : ""}`}
            title="Generate with .srt subtitles (sentence-level timestamps) for use in video editing"
          >
            <input
              checked={Boolean(subtitlesEnabled)}
              className="h-3.5 w-3.5 accent-violet-600"
              disabled={isBusy}
              onChange={(e) => onToggleSubtitles(e.target.checked)}
              type="checkbox"
            />
            <Captions className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Subtitles</span>
          </label>
        )}

        {onToggleStable && (
          <label
            className={`flex shrink-0 cursor-pointer select-none items-center gap-1.5 rounded-lg border px-2 py-1.5 font-semibold text-xs transition-colors ${
              stableEnabled
                ? "border-sky-500/60 bg-sky-500/10 text-sky-700 dark:text-sky-300"
                : "border-border text-muted-foreground hover:text-foreground"
            } ${isBusy ? "cursor-not-allowed opacity-50" : ""}`}
            title="Keep stable: the same text and voice will produce identical output every time. Disable for different intonation each time."
          >
            <input
              checked={Boolean(stableEnabled)}
              className="h-3.5 w-3.5 accent-sky-600"
              disabled={isBusy}
              onChange={(e) => onToggleStable(e.target.checked)}
              type="checkbox"
            />
            <Anchor className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Stable</span>
          </label>
        )}

        {/* Playback status and details. */}
        <div className="studio-audio-status flex flex-col">
          <div className="flex items-center gap-2">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                isGenerating
                  ? "animate-pulse bg-amber-500"
                  : isPlaying
                    ? "animate-pulse bg-brand-500"
                    : audioBlobUrl
                      ? "bg-brand-500"
                      : "bg-slate-400"
              }`}
            />
            <span className="font-semibold text-foreground text-xs capitalize sm:text-sm">
              {isGenerating
                ? "Generating speech..."
                : isPlaying
                  ? `Playing (${selectedVoiceName})`
                  : audioBlobUrl
                    ? `Generation complete (${selectedVoiceName})`
                    : `Ready (${selectedVoiceName})`}
            </span>
          </div>

          <span className="text-[12px] text-muted-foreground tabular-nums">
            {selectedVoiceName} ·{" "}
            {playbackSpeed.toLocaleString("en-US", {
              minimumFractionDigits: 2,
            })}
            × · WAV / MP3
          </span>
          {!isGenerating && ttfaMs !== null && (
            <div className="mt-0.5 flex items-center gap-1 font-mono font-semibold text-brand-600 text-xs dark:text-brand-400">
              <Zap className="h-3 w-3 fill-brand-500" />
              <span>
                Generated in:{" "}
                {ttfaMs >= 1000
                  ? `${(ttfaMs / 1000).toFixed(1)}s`
                  : `${ttfaMs}ms`}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Generation progress, seek bar, and waveform */}
      <div className="flex w-full min-w-0 flex-1 flex-col items-center justify-center space-y-1 px-2">
        {isGenerating && (
          <div className="w-full max-w-lg space-y-2">
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="font-medium text-muted-foreground">
                Estimated progress
              </span>
              <span className="font-bold font-mono text-brand-600 tabular-nums dark:text-brand-400">
                {progressPercent}%
              </span>
            </div>
            <div
              aria-label="Estimated speech generation progress"
              aria-valuemax={100}
              aria-valuemin={0}
              aria-valuenow={progressPercent}
              aria-valuetext={`Estimated ${progressPercent} percent`}
              className="h-3 w-full overflow-hidden rounded-full bg-brand-100 dark:bg-brand-950"
              role="progressbar"
            >
              <div
                className="h-full min-w-1 rounded-full bg-gradient-to-r from-brand-500 via-teal-500 to-cyan-500 transition-[width] duration-300 ease-out motion-reduce:transition-none"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-[12px] text-muted-foreground">
              <span>
                {progressPercent >= 95
                  ? "Still processing, please wait..."
                  : "Synthesizing audio..."}
              </span>
              <span className="font-mono tabular-nums">
                Elapsed {formatTime((progress?.elapsedMs ?? 0) / 1000)}
              </span>
            </div>
          </div>
        )}
        <canvas
          className={`${isGenerating ? "hidden" : ""} studio-wave h-[26px] w-full max-w-full rounded-lg px-1`}
          height={26}
          ref={canvasRef}
          width={240}
        />

        {/* Seek bar when audio is available */}
        {!isGenerating && audioBlobUrl && duration > 0 && (
          <div className="flex w-full max-w-full items-center gap-2 font-mono text-[12px] text-muted-foreground">
            <span>{formatTime(currentTime)}</span>
            <input
              className="h-1 flex-1 cursor-pointer appearance-none rounded-lg bg-secondary accent-brand-500"
              max={duration}
              min={0}
              onChange={(e) => handleSeek(Number.parseFloat(e.target.value))}
              step={0.1}
              type="range"
              value={currentTime}
            />
            <span>{formatTime(duration)}</span>
          </div>
        )}
      </div>

      {/* Playback, export, and volume controls */}
      <div className="flex w-full items-center justify-end gap-2 md:w-auto">
        <details className="studio-export" ref={exportRef}>
          <summary>Export file ▾</summary>
          <div className="studio-export-menu">
            {" "}
            {/* Download WAV */}
            <button
              className={`flex items-center gap-1 rounded-xl border px-3 py-2 font-semibold text-xs transition-all ${
                downloadable
                  ? "bg-secondary text-foreground shadow-sm hover:border-brand-500/50 hover:bg-brand-500/10 active:scale-95"
                  : "cursor-not-allowed border-transparent bg-muted text-muted-foreground opacity-40"
              }`}
              disabled={!downloadable}
              onClick={onDownload}
              title={
                downloadable ? "Download WAV file" : "No audio available to download"
              }
              type="button"
            >
              <Download className="h-3.5 w-3.5 text-brand-500" />
              <span>WAV</span>
            </button>
            {/* Download MP3 */}
            {onDownloadMp3 && (
              <button
                className={`flex items-center gap-1 rounded-xl border px-3 py-2 font-semibold text-xs transition-all ${
                  downloadable
                    ? "bg-secondary text-foreground shadow-sm hover:border-cyan-500/50 hover:bg-cyan-500/10 active:scale-95"
                    : "cursor-not-allowed border-transparent bg-muted text-muted-foreground opacity-40"
                }`}
                disabled={!downloadable}
                onClick={onDownloadMp3}
                title={
                  downloadable
                    ? "Download compressed MP3 file"
                    : "No audio available to download"
                }
                type="button"
              >
                <Download className="h-3.5 w-3.5 text-cyan-500" />
                <span>MP3</span>
              </button>
            )}
            {/* Download SRT */}
            {onDownloadSrt && (
              <button
                className="flex items-center gap-1 rounded-xl border bg-secondary px-3 py-2 font-semibold text-foreground text-xs shadow-sm transition-all hover:border-violet-500/50 hover:bg-violet-500/10 active:scale-95"
                onClick={onDownloadSrt}
                title="Download .srt subtitles (same name as the audio file)"
                type="button"
              >
                <Captions className="h-3.5 w-3.5 text-violet-500" />
                <span>SRT</span>
              </button>
            )}
          </div>
        </details>

        {/* Volume slider */}
        <div className="flex items-center gap-1.5 border-border/60 border-l pl-2">
          <button
            className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:text-foreground"
            onClick={() => onChangeVolume(volume === 0 ? 1 : 0)}
            title="Mute/Unmute"
            type="button"
          >
            {volume === 0 ? (
              <VolumeX className="h-3.5 w-3.5 text-rose-500" />
            ) : (
              <Volume2 className="h-3.5 w-3.5" />
            )}
          </button>
          <input
            className="h-1.5 w-14 cursor-pointer appearance-none rounded-lg bg-secondary accent-brand-500"
            max="1"
            min="0"
            onChange={(e) => onChangeVolume(Number.parseFloat(e.target.value))}
            step="0.05"
            title={`Volume: ${Math.round(volume * 100)}%`}
            type="range"
            value={volume}
          />
        </div>
      </div>
    </div>
  );
};
