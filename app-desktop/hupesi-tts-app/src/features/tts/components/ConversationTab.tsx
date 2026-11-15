import {
  AlertCircle,
  CheckCircle2,
  Download,
  Play,
  Sliders,
  Sparkles,
  Users,
} from "lucide-react";
import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { convertToMp3, synthesizeConversation } from "../api";
import { downloadBlobAsFile } from "../audio-engine";
import { LanguageSelect, useModelCaps } from "../model-caps";
import type { HistoryEntry, VoiceOption } from "../types";
import { groupVoices, voiceOptionLabel } from "../voice-label";
import { AdvancedGeneration, DEFAULT_TEMPERATURE } from "./AdvancedGeneration";

interface ConversationTabProps {
  /** Tab dang hien. Tab an van giu trong DOM nen phai tu dung audio khi bi an. */
  active: boolean;
  apiUrl: string;
  onAddHistory: (entry: HistoryEntry) => void;
  /** Kich ban khoi phuc tu lich su; doi `key` moi lan de nap lai ca kich ban cu. */
  restoredScript?: { key: number; script: string } | null;
  voices: VoiceOption[];
}

const SAMPLE_SCRIPTS = [
  {
    title: "AI Podcast Interview",
    script: `[Host]: [clears throat] Welcome back to the CIT Technology Podcast! Today we're joined by an AI expert.
[Guest]: Hello, Host, and hello to all our listeners. [laughs] I'm excited to share more about VieNeu-TTS v3 Turbo voice technology.
[Host]: Could you tell us about the biggest breakthrough in this version?
[Guest]: It's the ability to run in real time on a regular CPU and clone a voice in just a few seconds!`,
  },
  {
    title: "Story Dialogue",
    script: `[Narrator]: Long ago, in a forest filled with golden leaves, there lived a little white rabbit and a wise old turtle.
[Little Rabbit]: Mr. Turtle! [laughs] Would you like to race me to the top of the hill?
[Old Turtle]: [sighs] All right, little rabbit. But the road is long, and only patience will get you to the finish.`,
  },
  {
    title: "Customer Support",
    script: `[Agent]: Hupesi TTS, how can I help you?
[Customer]: [clears throat] Hi, I'd like to integrate AI voices into an app that automatically creates short videos.
[Agent]: [laughs] Absolutely! We offer a high-speed API that can export both studio-quality WAV and convenient compressed MP3 files.`,
  },
  {
    title: "Four-Person Panel",
    script: `[Host]: Welcome to our panel on digital transformation in education. Today we have three guests.
[Teacher]: Hello everyone. At my school, students are already used to submitting assignments and attending classes online.
[Parent]: [laughs] And as parents, we're happy that we can follow our children's progress right from our phones.
[Student]: My favorite part is being able to listen to lessons again with a generated voice while studying for exams.
[Host]: Thank you to all our guests. See you in the next episode!`,
  },
];

// English samples for multilingual models (Supertonic), using their own emotion tags.
const SAMPLE_SCRIPTS_FOREIGN = [
  {
    title: "AI Podcast Interview",
    script: `[Host]: Welcome back to the CIT Tech Podcast! Today we are joined by an AI expert.
[Guest]: Thanks for having me. <laugh> I'm happy to talk about AI voice technology.
[Host]: What is the biggest breakthrough in this new version?
[Guest]: It runs in real time on an ordinary computer, with no graphics card needed!`,
  },
  {
    title: "Story Dialogue",
    script: `[Narrator]: Long ago, in a forest full of golden leaves, there lived a little white rabbit and a wise old turtle.
[Rabbit]: Mister Turtle! <laugh> Do you want to race me to the top of the hill?
[Turtle]: <sigh> All right, little one. But the road is long, and only patience will get you there.`,
  },
  {
    title: "Customer Support",
    script: `[Agent]: Thank you for calling Hupesi TTS! How can I help you today?
[Customer]: Hi, I'd like to add an AI voice to my short video app.
[Agent]: <laugh> Of course! We have a fast API that exports both studio-quality WAV and compact MP3.`,
  },
  {
    title: "Four-Person Panel",
    script: `[Host]: Welcome to our panel on digital learning. Today we have three guests.
[Teacher]: Hello everyone. At my school, students now hand in homework and watch lessons online.
[Parent]: <laugh> And we parents can follow our children's progress right on our phones.
[Student]: My favorite part is listening to lessons again before exams.
[Host]: Thank you all. See you in the next episode!`,
  },
];

const TRAILING_SPACE = /\s+$/;

/** Returns an unused "Speaker N" name so the "Add speaker" button doesn't duplicate an existing one. */
function nextSpeakerName(existing: string[]): string {
  const taken = new Set(existing.map((name) => name.toLowerCase()));
  let n = existing.length + 1;
  while (taken.has(`speaker ${n}`)) {
    n += 1;
  }
  return `Speaker ${n}`;
}

const SPEAKER_COLORS = [
  "from-pink-500 to-rose-500",
  "from-cyan-500 to-blue-500",
  "from-brand-500 to-teal-500",
  "from-amber-500 to-orange-500",
  "from-purple-500 to-violet-500",
  "from-indigo-500 to-sky-500",
];

export const ConversationTab: React.FC<ConversationTabProps> = ({
  active,
  apiUrl,
  voices,
  onAddHistory,
  restoredScript,
}) => {
  const multilingual = useModelCaps().languages.length > 0;
  const samples = multilingual ? SAMPLE_SCRIPTS_FOREIGN : SAMPLE_SCRIPTS;
  const [script, setScript] = useState<string>(SAMPLE_SCRIPTS[0].script);

  // When the model changes, update the default sample script, but only if the user hasn't edited it.
  useEffect(() => {
    setScript((current) => {
      const isDefault =
        current === SAMPLE_SCRIPTS[0].script ||
        current === SAMPLE_SCRIPTS_FOREIGN[0].script;
      return isDefault ? samples[0].script : current;
    });
  }, [samples]);

  useEffect(() => {
    if (restoredScript) {
      setScript(restoredScript.script);
    }
  }, [restoredScript]);

  const scriptRef = useRef<HTMLTextAreaElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  useEffect(() => {
    if (!active) {
      audioRef.current?.pause();
    }
  }, [active]);
  const [speakerMapping, setSpeakerMapping] = useState<Record<string, string>>(
    {}
  );
  const [silenceDuration, setSilenceDuration] = useState<number>(0.4);
  const [speed, setSpeed] = useState<number>(1.0);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);

  // Generation state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [outputBlob, setOutputBlob] = useState<Blob | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState<number | null>(null);
  const [genDuration, setGenDuration] = useState<number | null>(null);
  const [turnsCount, setTurnsCount] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // The result URL is still used by history and the shared player when switching tabs.

  // Extract unique speakers from script
  const detectedSpeakers = useMemo(() => {
    const pattern = /^\s*(?:\[([^\]]+)\]|([^:]+))\s*:/gm;
    const speakers = new Set<string>();
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(script)) !== null) {
      const spk = (match[1] || match[2]).trim();
      if (spk) {
        speakers.add(spk);
      }
    }
    return Array.from(speakers);
  }, [script]);

  // Speakers whose voices were automatically replaced (old voice deleted / model changed):
  // notify the user so the podcast doesn't end up with the wrong voice unnoticed.
  const [replacedVoices, setReplacedVoices] = useState<Record<string, string>>(
    {}
  );
  const mappingRef = useRef(speakerMapping);
  mappingRef.current = speakerMapping;
  // Voice names from the previous list, used to report which voice was removed.
  const voiceNamesRef = useRef<Record<string, string>>({});

  // Assign voices to new speakers and reassign speakers whose voices are no longer available.
  useEffect(() => {
    if (voices.length === 0) {
      return;
    }
    const next = { ...mappingRef.current };
    const replaced: Record<string, string> = {};
    let changed = false;
    detectedSpeakers.forEach((spk, idx) => {
      if (voices.some((voice) => voice.id === next[spk])) {
        return;
      }
      // Distribute distinct voices
      const assignedVoice = voices[idx % voices.length];
      const oldId = next[spk];
      if (oldId) {
        const oldName = voiceNamesRef.current[oldId] ?? oldId;
        replaced[spk] =
          `The voice "${oldName}" for ${spk} is no longer available and was changed to "${assignedVoice.name}".`;
      }
      next[spk] = assignedVoice.id;
      changed = true;
    });
    voiceNamesRef.current = Object.fromEntries(
      voices.map((voice) => [voice.id, voice.name])
    );
    if (changed) {
      setSpeakerMapping(next);
    }
    if (Object.keys(replaced).length > 0) {
      setReplacedVoices((prev) => ({ ...prev, ...replaced }));
    }
  }, [detectedSpeakers, voices]);

  // Users may assume a conversation only has two people: add a new speaker line at the
  // end of the script and place the caret there so they can start typing immediately.
  const handleAddSpeaker = () => {
    const line = `[${nextSpeakerName(detectedSpeakers)}]: `;
    const trimmed = script.replace(TRAILING_SPACE, "");
    const next = trimmed ? `${trimmed}\n${line}` : line;
    setScript(next);
    requestAnimationFrame(() => {
      const el = scriptRef.current;
      if (el) {
        el.focus();
        el.setSelectionRange(next.length, next.length);
        el.scrollTop = el.scrollHeight;
      }
    });
  };

  // Handle Generate Conversation
  const handleGenerate = async () => {
    if (!script.trim()) {
      setErrorMessage("Please enter a conversation script!");
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);
    setReplacedVoices({});

    try {
      // Do not silently fall back to the default voice if a selected voice was deleted or the model changed.
      if (
        detectedSpeakers.some(
          (speaker) =>
            !voices.some((voice) => voice.id === speakerMapping[speaker])
        )
      ) {
        throw new Error("Please select a valid voice for each speaker.");
      }
      const validMapping = Object.fromEntries(
        Object.entries(speakerMapping).filter(([, id]) =>
          voices.some((v) => v.id === id)
        )
      );
      const res = await synthesizeConversation(apiUrl, {
        script,
        speakerMapping: validMapping,
        silenceDuration,
        speed,
        temperature,
        format: "wav",
      });

      setOutputBlob(res.blob);
      const url = URL.createObjectURL(res.blob);
      setOutputUrl(url);
      setElapsedSec(res.elapsedSeconds);
      setGenDuration(res.durationSeconds);
      setTurnsCount(res.turnsCount || detectedSpeakers.length);

      onAddHistory({
        id: `conv_${Date.now()}`,
        text: script,
        voiceId: "multi_speaker",
        voiceName: `Conversation (${detectedSpeakers.join(", ")})`,
        timestamp: Date.now(),
        durationSeconds: res.durationSeconds,
        audioBlobUrl: url,
        audioBlob: res.blob,
        mode: "conversation",
        format: "wav",
      });
    } catch (err: any) {
      setErrorMessage(err.message || "Error generating conversation audio.");
    } finally {
      setIsGenerating(false);
    }
  };

  // Download WAV
  const handleDownloadWav = async () => {
    if (!outputBlob) {
      return;
    }
    await downloadBlobAsFile(outputBlob, `cit_podcast_${Date.now()}.wav`);
  };

  // Download MP3
  const handleDownloadMp3 = async () => {
    if (!outputBlob) {
      return;
    }
    try {
      const mp3Blob = await convertToMp3(apiUrl, outputBlob);
      await downloadBlobAsFile(mp3Blob, `cit_podcast_${Date.now()}.mp3`);
    } catch {
      handleDownloadWav();
    }
  };

  return (
    <div className="studio-tab-grid studio-conversation-layout fade-in-50 animate-in duration-300">
      {/* Left Column: Script Editor (7 cols) */}
      <div className="studio-span-7 flex flex-col space-y-4">
        <div className="studio-tab-card flex flex-1 flex-col space-y-3 rounded-[14px] border border-border bg-card p-5 shadow-sm">
          <div className="studio-section-heading flex items-center justify-between border-border/60 border-b pb-3">
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5 text-brand-500" />
              <h3 className="font-bold text-foreground text-sm">
                Conversation / Podcast Script
              </h3>
            </div>

            {/* Character count & detected speakers count */}
            <div className="flex items-center gap-2 font-mono text-muted-foreground text-xs">
              <span>{detectedSpeakers.length} speakers</span>
              <span>·</span>
              <span>{script.length} characters</span>
            </div>
          </div>

          {/* Preset Sample Scripts */}
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="flex items-center gap-1 text-[12px] text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5 text-amber-500" />
              Sample scripts:
            </span>
            {samples.map((item) => (
              <button
                className="rounded-full border border-border/60 bg-secondary/80 px-2.5 py-1 text-foreground/80 text-xs transition-all hover:bg-secondary hover:text-foreground active:scale-95"
                disabled={isGenerating}
                key={item.title}
                onClick={() => setScript(item.script)}
                type="button"
              >
                {item.title}
              </button>
            ))}
          </div>

          {/* Script Textarea */}
          <div className="studio-conversation-editor relative flex flex-col pt-1">
            <textarea
              className="w-full flex-1 resize-none rounded-xl border border-border/80 bg-secondary/20 p-3.5 font-sans text-[14px] text-foreground leading-relaxed outline-none focus:border-brand-500"
              disabled={isGenerating}
              onChange={(e) => setScript(e.target.value)}
              placeholder={
                "Enter a conversation script using this format:\n[Speaker 1]: Speaker 1's dialogue...\n[Speaker 2]: Speaker 2's dialogue...\n[Speaker 3]: Speaker 3's dialogue...\nYou can add as many speakers as you like."
              }
              ref={scriptRef}
              value={script}
            />
          </div>

          <div className="flex items-center justify-between gap-2 text-[12px] text-muted-foreground">
            <span>
              Each new name in <code>[Name]:</code> becomes a speaker, with no
              limit on the number of speakers.
            </span>
            <button
              className="flex-shrink-0 rounded-full border border-border/60 bg-secondary/80 px-2.5 py-1 font-medium text-foreground/80 text-xs transition-all hover:bg-secondary hover:text-foreground active:scale-95 disabled:opacity-50"
              disabled={isGenerating}
              onClick={handleAddSpeaker}
              type="button"
            >
              + Add speaker
            </button>
          </div>

          {/* Trigger Generate Button */}
          <div className="flex items-center gap-3 pt-2">
            <button
              className={`flex flex-1 select-none items-center justify-center gap-2.5 rounded-xl px-6 py-3 font-bold text-sm shadow-md transition-all active:scale-95 ${
                isGenerating
                  ? "animate-pulse bg-amber-500 text-white"
                  : "studio-primary"
              } ${script.trim() ? "" : "cursor-not-allowed opacity-50"}`}
              disabled={isGenerating || !script.trim()}
              onClick={handleGenerate}
              type="button"
            >
              <Play className="h-4 w-4 fill-white" />
              <span>
                {isGenerating
                  ? "Processing and stitching audio segments..."
                  : "Generate Conversation / Podcast Audio"}
              </span>
            </button>
          </div>

          {/* Error message */}
          {errorMessage && (
            <div className="flex items-center gap-2 rounded-xl border border-rose-500/20 bg-rose-500/10 p-3 text-rose-600 text-xs dark:text-rose-400">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}
        </div>
      </div>

      {/* Right Column: Speaker Mapping & Podcast Parameters (5 cols) */}
      <div className="studio-span-5 flex flex-col space-y-4">
        {/* Step 2: Speaker Voice Assign Box */}
        <div className="studio-tab-card space-y-4 rounded-[14px] border border-border bg-card p-5 shadow-sm">
          <div className="studio-section-heading flex items-center justify-between border-border/60 border-b pb-3">
            <div className="flex items-center gap-2">
              <Sliders className="h-4 w-4 text-brand-500" />
              <h3 className="font-bold text-foreground text-sm">
                Speaker Voice Assignment
              </h3>
            </div>
            <span className="text-muted-foreground text-xs">
              {detectedSpeakers.length} speakers
            </span>
          </div>

          {detectedSpeakers.some((spk) => replacedVoices[spk]) && (
            <div
              className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-amber-700 text-xs dark:text-amber-400"
              role="status"
            >
              <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <div className="space-y-1">
                {detectedSpeakers
                  .filter((spk) => replacedVoices[spk])
                  .map((spk) => (
                    <p key={spk}>{replacedVoices[spk]}</p>
                  ))}
                <p className="font-medium">Please review before generating.</p>
              </div>
            </div>
          )}

          {/* Detected Speakers List */}
          <div className="max-h-[220px] space-y-2.5 overflow-y-auto pr-1">
            {detectedSpeakers.length === 0 ? (
              <div className="rounded-xl border border-border border-dashed bg-secondary/30 p-4 text-center text-muted-foreground text-xs">
                No speakers detected. Write the script in the format{" "}
                <code>[Name]: Dialogue</code>.
              </div>
            ) : (
              detectedSpeakers.map((spk, idx) => {
                const color = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
                const mapped = speakerMapping[spk];
                const currentVoiceId =
                  mapped && voices.some((v) => v.id === mapped)
                    ? mapped
                    : voices[0]?.id || "";

                return (
                  <div
                    className="studio-section-heading flex items-center justify-between gap-3 rounded-xl border border-border/60 bg-secondary/30 p-3"
                    key={spk}
                  >
                    <div className="flex min-w-0 items-center gap-2.5">
                      <div
                        className={`h-7 w-7 rounded-lg bg-gradient-to-tr ${color} flex flex-shrink-0 items-center justify-center font-bold text-white text-xs shadow-sm`}
                      >
                        {spk.slice(0, 1).toUpperCase()}
                      </div>
                      <span className="truncate font-semibold text-foreground text-xs">
                        {spk}
                      </span>
                    </div>

                    {/* Voice Select */}
                    <select
                      className="max-w-[260px] rounded-lg border border-border/80 bg-background px-2.5 py-1.5 text-xs outline-none focus:border-brand-500"
                      onChange={(e) => {
                        setSpeakerMapping((prev) => ({
                          ...prev,
                          [spk]: e.target.value,
                        }));
                        setReplacedVoices((prev) => {
                          const { [spk]: _, ...rest } = prev;
                          return rest;
                        });
                      }}
                      value={currentVoiceId}
                    >
                      {groupVoices(voices).map((g) => (
                        <optgroup key={g.label} label={g.label}>
                          {g.voices.map((v) => (
                            <option key={v.id} value={v.id}>
                              {voiceOptionLabel(v)}
                            </option>
                          ))}
                        </optgroup>
                      ))}
                    </select>
                  </div>
                );
              })
            )}
          </div>

          {/* Podcast Parameters */}
          <div className="space-y-3 border-border/60 border-t pt-3">
            <LanguageSelect />
            {/* Silence interval between dialogue turns */}
            <div>
              <div className="mb-1 flex justify-between font-medium text-foreground text-xs">
                <span>Pause between lines:</span>
                <span className="font-bold font-mono text-brand-600 dark:text-brand-400">
                  {silenceDuration.toFixed(2)}s
                </span>
              </div>
              <input
                className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-secondary accent-brand-500"
                max="1.5"
                min="0.1"
                onChange={(e) =>
                  setSilenceDuration(Number.parseFloat(e.target.value))
                }
                step="0.05"
                type="range"
                value={silenceDuration}
              />
              <div className="mt-0.5 flex justify-between text-[12px] text-muted-foreground">
                <span>0.1s (Fast)</span>
                <span>0.4s (Natural)</span>
                <span>1.5s (Slow)</span>
              </div>
            </div>

            {/* Speed */}
            <div className="pt-1">
              <div>
                <div className="mb-1 flex justify-between text-[12px] text-muted-foreground">
                  <span>Speed:</span>
                  <span className="font-bold font-mono text-foreground">
                    {speed.toFixed(2)}x
                  </span>
                </div>
                <input
                  className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-secondary accent-brand-500"
                  max="1.5"
                  min="0.75"
                  onChange={(e) => setSpeed(Number.parseFloat(e.target.value))}
                  step="0.05"
                  type="range"
                  value={speed}
                />
              </div>
            </div>

            <AdvancedGeneration
              disabled={isGenerating}
              onChangeTemperature={setTemperature}
              temperature={temperature}
            />
          </div>
        </div>

        {/* Output Result Box */}
        {outputUrl && (
          <div className="slide-in-from-bottom-3 animate-in space-y-3 rounded-2xl border border-brand-500/40 bg-card p-5 shadow-sm">
            <div className="studio-section-heading flex items-center justify-between border-border/60 border-b pb-2">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-brand-500" />
                <h4 className="font-bold text-foreground text-xs">
                  Podcast complete
                </h4>
              </div>
              <span className="font-mono text-[12px] text-muted-foreground">
                {turnsCount} turns · Completed in {elapsedSec?.toFixed(1)}s ·{" "}
                {genDuration?.toFixed(1)}s audio
              </span>
            </div>

            {/* Audio Player */}
            <audio
              className="h-10 w-full"
              controls
              ref={audioRef}
              src={outputUrl}
            />

            {/* Export Buttons */}
            <div className="flex items-center gap-2 pt-1">
              <button
                className="flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-border bg-secondary px-3 py-2 font-medium text-foreground text-xs shadow-sm transition-colors hover:bg-secondary/80"
                onClick={handleDownloadWav}
                type="button"
              >
                <Download className="h-3.5 w-3.5 text-brand-500" />
                <span>Download WAV</span>
              </button>
              <button
                className="flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-border bg-secondary px-3 py-2 font-medium text-foreground text-xs shadow-sm transition-colors hover:bg-secondary/80"
                onClick={handleDownloadMp3}
                type="button"
              >
                <Download className="h-3.5 w-3.5 text-cyan-500" />
                <span>Download MP3</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
