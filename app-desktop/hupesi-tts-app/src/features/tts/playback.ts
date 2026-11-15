/** Play or pause the shared audio player; restart from the beginning if playback has ended. */
export async function toggleAudioPlayback(
  audio: HTMLAudioElement
): Promise<void> {
  if (!(audio.paused || audio.ended)) {
    audio.pause();
    return;
  }
  if (audio.ended) {
    audio.currentTime = 0;
  }
  try {
    await audio.play();
  } catch (error) {
    // Switching tracks or stopping quickly can cancel a pending play request.
    if (!(error instanceof Error && error.name === "AbortError")) {
      console.warn("Unable to play audio:", error);
    }
  }
}

/** Applies to both sample playback and result playback: only one player may play at a time. */
export function pauseOtherAudio(
  active: HTMLAudioElement,
  players: Iterable<HTMLAudioElement>
): void {
  // Ignore stale play events that may be queued after quickly switching tracks.
  if (active.paused) {
    return;
  }
  for (const player of players) {
    if (player !== active && !player.paused) {
      player.pause();
    }
  }
}
