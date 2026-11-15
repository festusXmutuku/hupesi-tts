// Reference voice sample for cloning: the engine uses at most 8 seconds of speech
// (after trimming silence at both ends). Keeping 15 seconds provides headroom, so
// hour-long recordings don't need to be uploaded in full (60 minutes = ~330 MB WAV)
// just to extract the first few seconds.
export const REF_MAX_SECONDS = 15;

const VOICE_THRESHOLD = 0.01; // ~ -40 dBFS: below this level is treated as silence/background noise
const LEAD_SECONDS = 0.1; // keep a small amount of silence before the first speech sample

/**
 * ``[start, end)`` range to keep: starts just before the first voiced sample and
 * lasts at most ``maxSeconds``. If the audio is entirely silent, start from the beginning.
 */
export function voicedWindow(
  samples: Float32Array,
  sampleRate: number,
  maxSeconds: number = REF_MAX_SECONDS
): [number, number] {
  let first = 0;
  while (
    first < samples.length &&
    Math.abs(samples[first]) <= VOICE_THRESHOLD
  ) {
    first++;
  }
  if (first >= samples.length) {
    first = 0;
  }
  const start = Math.max(0, first - Math.round(LEAD_SECONDS * sampleRate));
  const end = Math.min(
    samples.length,
    start + Math.round(maxSeconds * sampleRate)
  );
  return [start, end];
}
