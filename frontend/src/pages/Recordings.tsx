import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listRecordings } from "../api";
import StatusBadge from "../components/StatusBadge";
import { formatDate, formatDuration } from "../lib/format";
import { isInFlight } from "../lib/transcript";
import type { Recording } from "../types";

const POLL_MS = 3000;

export default function Recordings() {
  const [recs, setRecs] = useState<Recording[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load once, then keep polling only while something is still in flight.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function load() {
      try {
        const data = await listRecordings();
        if (cancelled) return;
        setRecs(data);
        setError(null);
        if (data.some((r) => isInFlight(r.status))) timer = setTimeout(load, POLL_MS);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Could not load recordings.");
        timer = setTimeout(load, POLL_MS * 3);
      }
    }
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  return (
    <>
      <h1>Recordings</h1>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {recs === null && !error && <p className="muted">Loading…</p>}
      {recs !== null && recs.length === 0 && <p className="muted">No recordings yet.</p>}
      {recs !== null && recs.length > 0 && (
        <div className="card flush">
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Date</th>
                <th>Duration</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {recs.map((r) => (
                <tr key={r.id}>
                  <td>
                    <Link to={`/recordings/${r.id}`}>{r.original_filename}</Link>
                    {r.reference && <div className="muted small">{r.reference}</div>}
                  </td>
                  <td>{formatDate(r.created_at)}</td>
                  <td>{formatDuration(r.duration_sec)}</td>
                  <td>
                    <StatusBadge rec={r} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
