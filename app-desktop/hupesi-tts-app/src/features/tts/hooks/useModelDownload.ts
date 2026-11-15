import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelModelDownload,
  fetchDownloadStatus,
  fetchModels,
  startModelDownload,
  switchModel,
} from "../api";
import type { DownloadState, ModelOption } from "../types";

const EMPTY_STATE: DownloadState = {
  state: "idle",
  model_id: null,
  percent: 0,
  done_bytes: 0,
  total_bytes: 0,
  current_file: "",
  message: "",
};

/** Poll interval while downloading. Smooth enough for progress, cheap enough for the backend. */
const POLL_INTERVAL_MS = 1000;

/**
 * State machine for downloading & applying models.
 *
 * Downloads run in a background thread on the server, so closing the modal doesn't stop them -
 * reopening will render the correct progress on the first poll. This hook stores nothing locally.
 */
export function useModelDownload(apiUrl: string, enabled: boolean) {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [status, setStatus] = useState<DownloadState>(EMPTY_STATE);
  const [busy, setBusy] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetchModels(apiUrl);
      if (res.models) {
        setModels(res.models);
      }
      if (res.current?.name) {
        setCurrent(res.current.name);
      }
    } catch {
      // Server not ready yet is normal when the app has just started.
    }
  }, [apiUrl]);

  const stopPolling = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    if (timerRef.current !== null) {
      return;
    }
    timerRef.current = setInterval(async () => {
      try {
        const st = await fetchDownloadStatus(apiUrl);
        setStatus(st);
        if (st.state !== "downloading" && st.state !== "applying") {
          stopPolling();
          setBusy(false);
          setCancelling(false);
          setError(st.state === "error" ? st.message : null);
          await refresh(); // "Not downloaded" badge must change to "Downloaded"
        }
      } catch {
        stopPolling();
        setBusy(false);
      }
    }, POLL_INTERVAL_MS);
  }, [apiUrl, stopPolling, refresh]);

  // On open: load the list and resume polling if the server is mid-download.
  useEffect(() => {
    if (!enabled) {
      return;
    }
    let cancelled = false;

    void refresh();
    void (async () => {
      try {
        const st = await fetchDownloadStatus(apiUrl);
        if (cancelled) {
          return;
        }
        setStatus(st);
        if (st.state === "downloading" || st.state === "applying") {
          setBusy(true);
          startPolling();
        }
      } catch {
        // skip
      }
    })();

    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [enabled, apiUrl, refresh, startPolling, stopPolling]);

  const download = useCallback(
    async (modelId: string, pack: string | null = null) => {
      setError(null);
      setCancelling(false);
      setBusy(true);
      setStatus({
        ...EMPTY_STATE,
        state: "downloading",
        model_id: modelId,
        pack,
      });
      try {
        await startModelDownload(apiUrl, modelId, true, pack);
        startPolling();
      } catch (e) {
        setBusy(false);
        const message = e instanceof Error ? e.message : String(e);
        setStatus({
          ...EMPTY_STATE,
          state: "error",
          model_id: modelId,
          message,
        });
        setError(message);
      }
    },
    [apiUrl, startPolling]
  );

  const apply = useCallback(
    async (modelId: string) => {
      setError(null);
      setCancelling(false);
      setBusy(true);
      setStatus({ ...EMPTY_STATE, state: "applying", model_id: modelId });
      try {
        await switchModel(apiUrl, modelId, "auto");
        setStatus({
          ...EMPTY_STATE,
          state: "ready",
          model_id: modelId,
          message: "Model applied. You can start generating speech.",
        });
        await refresh();
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        setStatus({
          ...EMPTY_STATE,
          state: "error",
          model_id: modelId,
          message,
        });
        setError(message);
      } finally {
        setBusy(false);
      }
    },
    [apiUrl, refresh]
  );

  const cancel = useCallback(async () => {
    setCancelling(true);
    try {
      await cancelModelDownload(apiUrl);
    } catch (e) {
      setCancelling(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [apiUrl]);

  return {
    models,
    current,
    status,
    busy,
    cancelling,
    error,
    download,
    apply,
    cancel,
    refresh,
  };
}
