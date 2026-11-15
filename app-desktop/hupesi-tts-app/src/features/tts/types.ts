export interface VoiceOption {
  avatar?: string;
  description?: string;
  /** Voice selected by the VieNeu author (shown at the top of the list). */
  featured?: boolean;
  gender?: "male" | "female" | "neutral";
  id: string;
  isCloned?: boolean;
  /** Language the voice was trained for (Kokoro: "en", "fr", etc.); not available for VieNeu/Supertonic. */
  language?: string;
  name: string;
  region?: "North" | "South" | "Central" | "Standard" | null;
}

export type PlaybackStatus =
  | "idle"
  | "generating"
  | "streaming"
  | "playing"
  | "paused"
  | "error";

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export type ActiveTab =
  | "tts"
  | "clone"
  | "conversation"
  | "pdf"
  | "batch"
  | "history"
  | "about";

export interface HistoryEntry {
  /** Co san khi vua tao; muc nap lai tu o dia thi tai ve khi can (xem history.ts). */
  audioBlob?: Blob;
  /** blob: URL khi vua tao, hoac URL cua server voi muc da luu. */
  audioBlobUrl: string;
  durationSeconds: number;
  format?: "wav" | "mp3";
  id: string;
  mode?: "tts" | "clone" | "conversation" | "pdf";
  /** Id cua muc tren server (lich su luu tren dia); chua co khi dang luu / luu loi. */
  serverId?: string;
  /** Noi dung phu de SRT (khi tao kem phu de). */
  subtitles?: string;
  /** URL tai phu de cua muc da luu tren server. */
  subtitlesUrl?: string;
  text: string;
  timestamp: number;
  voiceId: string;
  voiceName: string;
}

export interface AppSettings {
  apiUrl: string;
  autoPlayOnGenerate: boolean;
  /** Default voice when opening the app. Empty = the model's default voice. */
  defaultVoiceId?: string;
  modelName?: string;
  playbackSpeed: number;
  sampleRate: number;
  silenceDuration?: number;
}

export interface DialogueTurn {
  id: string;
  speaker: string;
  text: string;
  voiceId?: string;
}

export interface ModelOption {
  /** Estimated download size, shown as "Not downloaded - 453 MB". */
  approx_bytes?: number;
  /** Whether this machine can run the model. Determined automatically by the backend (CUDA, torch, etc.). */
  available?: boolean;
  /** Whether this machine can run the model. Separate from whether it is downloaded. */
  compatible?: boolean;
  description: string;
  device: string;
  /** Whether the model files are already fully present in the local Hugging Face cache. */
  downloaded?: boolean;
  id: string;
  name: string;
  /** Model downloaded but the audio separation add-on is not installed yet (Kokoro). */
  needs_addon_setup?: boolean;
  needs_gpu_setup?: boolean;
  /** Additional note, for example FP32 requires downloading ~453 MB more on first selection. */
  note?: string | null;
  /** Additional downloadable language packs (Kokoro: Japanese, Chinese). */
  packs?: LanguagePack[];
  precision: string;
  supports_cloning: boolean;
  supports_streaming: boolean;
  /** Reason the model cannot be used, shown to the user instead of failing unexpectedly. */
  unavailable_reason?: string | null;
}

export interface PdfExtractionResult {
  filename: string;
  page_count: number;
  pages: string[];
  text: string;
}

/** Additional downloadable language pack for a model (Kokoro: Japanese, Chinese). */
export interface LanguagePack {
  approx_bytes: number;
  code: string;
  installed: boolean;
  name: string;
}

/** Progress of a model download running in a background server thread. */
export interface DownloadState {
  current_file: string;
  done_bytes: number;
  message: string;
  model_id: string | null;
  /** Language pack being downloaded ("ja"/"zh"); null = downloading the full model. */
  pack?: string | null;
  percent: number;
  phase?: "model" | "gpu_setup" | "addon_setup";
  state: "idle" | "downloading" | "applying" | "ready" | "error" | "cancelled";
  total_bytes: number;
}
