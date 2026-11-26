import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  downloadTranscriptJson,
  fetchAudio,
  getRecording,
  getTranscript,
  retryRecording,
} from "../api";
import { useAuth } from "../auth";
import StatusBadge from "../components/StatusBadge";
import TranscriptView from "../components/TranscriptView";
import { formatDate, formatDuration } from "../lib/format";
import { isInFlight } from "../lib/transcript";
import type { Recording, Transcript } from "../types";

const POLL_MS = 2000;

export default function TranscriptPage() {
  const { id = "" } = useParams();
  const { session } = useAuth();
  const isAdmin = session?.role === "admin";

  const [rec, setRec] = useState<Recording | null>(null);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [reloadKey, setReloadKey] = useState(0); // bump to restart polling (e.g. after a retry)
  const audio = useRef<HTMLAudioElement>(null);

  // Poll the recording until it reaches a final state.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function load() {
      try {
        const r = await getRecording(id);
        if (cancelled) return;
        setRec(r);
        setError(null);
        if (isInFlight(r.status)) timer = setTimeout(load, POLL_MS);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load the recording.");
      }
    }
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [id, reloadKey]);

  // Once completed, fetch the transcript the user's role allows (admins: original text).
  const status = rec?.status;
  useEffect(() => {
    if (status !== "completed") {
      setTranscript(null);
      return;
    }
    let cancelled = false;
    getTranscript(id, isAdmin)
      .then((t) => !cancelled && setTranscript(t))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Could not load the transcript."));
    return () => {
      cancelled = true;
    };
  }, [id, status, isAdmin]);

  // Audio is admin-only and protected, so it is fetched with the token and played from a blob URL.
  useEffect(() => {
    if (status !== "completed" || !isAdmin) return;
    let cancelled = false;
    let url: string | null = null;
    fetchAudio(id)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setAudioUrl(url);
      })
      .catch(() => undefined); // the transcript is still usable without playback
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
      setAudioUrl(null);
    };
  }, [id, status, isAdmin]);

  const seek = useCallback((sec: number) => {
    const el = audio.current;
    if (!el) return;
    el.currentTime = sec;
    void el.play().catch(() => undefined);
  }, []);

  async function onDownload() {
    setBusy(true);
    try {
      await downloadTranscriptJson(id, isAdmin);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed.");
    } finally {
      setBusy(false);
    }
  }

  async function onRetry() {
    setBusy(true);
    try {
      setRec(await retryRecording(id));
      setError(null);
      setReloadKey((k) => k + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Retry failed.");
    } finally {
      setBusy(false);
    }
  }

  const pendingRedaction = transcript && !isAdmin && transcript.redaction !== "available";

  return (
    <>
      <p>
        <Link to="/">← Recordings</Link>
      </p>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {!rec && !error && <p className="muted">Loading…</p>}
      {rec && (
        <>
          <div className="row between">
            <div>
              <h1>{rec.original_filename}</h1>
              <p className="muted">
                {formatDate(rec.created_at)} · {formatDuration(rec.duration_sec)} ·{" "}
                {transcript ? `${transcript.speaker_count} speaker${transcript.speaker_count === 1 ? "" : "s"}` : "-"}
              </p>
            </div>
            <div className="row">
              <StatusBadge rec={rec} />
              {rec.status === "completed" && (
                <button type="button" onClick={onDownload} disabled={busy}>
                  Download JSON
                </button>
              )}
              {isAdmin && rec.status === "failed" && rec.sha256 && (
                <button type="button" onClick={onRetry} disabled={busy}>
                  Retry
                </button>
              )}
            </div>
          </div>

          {isInFlight(rec.status) && <p className="notice">This recording is being processed. This page updates automatically.</p>}
          {rec.status === "failed" && <p className="error">{rec.error ?? "Processing failed."}</p>}

          {isAdmin && audioUrl && (
            <audio
              ref={audio}
              className="player"
              controls
              src={audioUrl}
              onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
            />
          )}

          {pendingRedaction && (
            <p className="notice">
              The redacted transcript is not available yet. Text is withheld until PHI redaction is in place.
            </p>
          )}
          {transcript && (
            <div className="card">
              <TranscriptView
                segments={transcript.segments}
                currentTime={isAdmin && audioUrl ? time : undefined}
                playing={playing}
                onSeek={isAdmin && audioUrl ? seek : undefined}
              />
            </div>
          )}
        </>
      )}
    </>
  );
}
