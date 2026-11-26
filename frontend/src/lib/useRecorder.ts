import { useCallback, useEffect, useRef, useState } from "react";
import { pickRecorderFormat } from "./mime";

export type RecState = "idle" | "recording" | "paused" | "stopped";

export interface RecordedAudio {
  file: File;
  url: string; // object URL for the preview player
}

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/** Microphone recording is only available on secure origins (HTTPS or localhost). */
export function recordingSupport(): { ok: boolean; reason?: string } {
  if (typeof window === "undefined") return { ok: false, reason: "Not available." };
  if (!window.isSecureContext) {
    return {
      ok: false,
      reason: "Recording from the microphone needs a secure connection (HTTPS). File upload still works.",
    };
  }
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    return { ok: false, reason: "This browser does not support recording." };
  }
  return { ok: true };
}

export function useRecorder() {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [state, setState] = useState<RecState>("idle");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [level, setLevel] = useState(0);
  const [result, setResult] = useState<RecordedAudio | null>(null);
  const [error, setError] = useState<string | null>(null);

  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);
  const raf = useRef<number | null>(null);
  const chunks = useRef<Blob[]>([]);
  const accumulated = useRef(0);
  const segmentStart = useRef(0);
  const resultUrl = useRef<string | null>(null);

  const releaseCapture = useCallback(() => {
    if (raf.current !== null) cancelAnimationFrame(raf.current);
    raf.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    void audioCtx.current?.close().catch(() => undefined);
    audioCtx.current = null;
    setLevel(0);
  }, []);

  const revokeResult = useCallback(() => {
    if (resultUrl.current) URL.revokeObjectURL(resultUrl.current);
    resultUrl.current = null;
  }, []);

  useEffect(
    () => () => {
      if (recorder.current && recorder.current.state !== "inactive") {
        recorder.current.onstop = null;
        recorder.current.stop();
      }
      releaseCapture();
      revokeResult();
    },
    [releaseCapture, revokeResult],
  );

  // Timer: counts only the time spent actually recording (pauses are excluded).
  useEffect(() => {
    if (state !== "recording") return;
    const id = setInterval(() => setElapsedMs(accumulated.current + (Date.now() - segmentStart.current)), 200);
    return () => clearInterval(id);
  }, [state]);

  /** Ask for permission once so the browser reveals device names, then list the microphones. */
  const enableMic = useCallback(async () => {
    setError(null);
    try {
      const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
      probe.getTracks().forEach((t) => t.stop());
      const inputs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
      setDevices(inputs);
      setDeviceId((cur) => cur || inputs[0]?.deviceId || "");
    } catch (e) {
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Microphone access was blocked. Allow it in the browser address bar and try again."
          : "Could not access a microphone.",
      );
    }
  }, []);

  const start = useCallback(async () => {
    setError(null);
    const format = pickRecorderFormat((m) => MediaRecorder.isTypeSupported(m));
    if (!format) {
      setError("This browser cannot record in a supported audio format.");
      return;
    }
    try {
      revokeResult();
      setResult(null);
      const s = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          // Cleanup is done server-side (DeepFilterNet), so keep the browser's processing minimal.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });
      stream.current = s;

      const ctx = new AudioContext();
      audioCtx.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      ctx.createMediaStreamSource(s).connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      const tick = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) sum += ((v - 128) / 128) ** 2;
        setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 3));
        raf.current = requestAnimationFrame(tick);
      };
      tick();

      chunks.current = [];
      const rec = new MediaRecorder(s, { mimeType: format.mime });
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };
      rec.onstop = () => {
        const blob = new Blob(chunks.current, { type: format.mime.split(";")[0] });
        const file = new File([blob], `recording-${stamp()}.${format.ext}`, { type: blob.type });
        const url = URL.createObjectURL(blob);
        resultUrl.current = url;
        setResult({ file, url });
        releaseCapture();
        setState("stopped");
      };
      recorder.current = rec;
      accumulated.current = 0;
      segmentStart.current = Date.now();
      setElapsedMs(0);
      rec.start(1000);
      setState("recording");
    } catch (e) {
      releaseCapture();
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Microphone access was blocked."
          : "Could not start recording.",
      );
    }
  }, [deviceId, releaseCapture, revokeResult]);

  const pause = useCallback(() => {
    if (recorder.current?.state !== "recording") return;
    recorder.current.pause();
    accumulated.current += Date.now() - segmentStart.current;
    setElapsedMs(accumulated.current);
    setState("paused");
  }, []);

  const resume = useCallback(() => {
    if (recorder.current?.state !== "paused") return;
    recorder.current.resume();
    segmentStart.current = Date.now();
    setState("recording");
  }, []);

  const stop = useCallback(() => {
    const rec = recorder.current;
    if (!rec || rec.state === "inactive") return;
    if (rec.state === "recording") accumulated.current += Date.now() - segmentStart.current;
    setElapsedMs(accumulated.current);
    rec.stop();
  }, []);

  const discard = useCallback(() => {
    revokeResult();
    setResult(null);
    setElapsedMs(0);
    setState("idle");
  }, [revokeResult]);

  return { devices, deviceId, setDeviceId, state, elapsedMs, level, result, error, enableMic, start, pause, resume, stop, discard };
}
