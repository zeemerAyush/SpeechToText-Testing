import { useEffect } from "react";
import { formatDuration } from "../lib/format";
import { recordingSupport, useRecorder } from "../lib/useRecorder";

/** Browser microphone recorder. Reports the finished recording (or null) through onFile. */
export default function Recorder({ onFile }: { onFile: (file: File | null) => void }) {
  const support = recordingSupport();
  const r = useRecorder();

  useEffect(() => {
    onFile(r.result?.file ?? null);
  }, [r.result, onFile]);

  if (!support.ok) return <p className="notice">{support.reason}</p>;

  const active = r.state === "recording" || r.state === "paused";

  return (
    <div className="recorder">
      {r.devices.length === 0 ? (
        <button type="button" onClick={() => void r.enableMic()}>
          Enable microphone
        </button>
      ) : (
        <label>
          Microphone
          <select value={r.deviceId} onChange={(e) => r.setDeviceId(e.target.value)} disabled={active}>
            {r.devices.map((d, i) => (
              <option key={d.deviceId || i} value={d.deviceId}>
                {d.label || `Microphone ${i + 1}`}
              </option>
            ))}
          </select>
        </label>
      )}

      <div className="meter" role="meter" aria-label="Input level" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(r.level * 100)}>
        <div className="meter-fill" style={{ width: `${Math.round(r.level * 100)}%` }} />
      </div>
      <div className="timer" aria-live="off">
        {r.state === "recording" && <span className="rec-dot" aria-hidden="true" />}
        {formatDuration(r.elapsedMs / 1000)}
        {r.state === "paused" && <span className="muted small"> paused</span>}
      </div>

      <div className="row">
        {(r.state === "idle" || r.state === "stopped") && (
          <button type="button" className="primary" onClick={() => void r.start()} disabled={r.devices.length === 0}>
            {r.state === "stopped" ? "Record again" : "Start"}
          </button>
        )}
        {r.state === "recording" && (
          <button type="button" onClick={r.pause}>
            Pause
          </button>
        )}
        {r.state === "paused" && (
          <button type="button" onClick={r.resume}>
            Resume
          </button>
        )}
        {active && (
          <button type="button" className="danger" onClick={r.stop}>
            Stop
          </button>
        )}
        {r.state === "stopped" && (
          <button type="button" onClick={r.discard}>
            Discard
          </button>
        )}
      </div>

      {r.result && (
        <div className="preview">
          <p className="muted small">Preview ({r.result.file.name})</p>
          <audio controls src={r.result.url} />
        </div>
      )}
      {r.error && (
        <p role="alert" className="error">
          {r.error}
        </p>
      )}
    </div>
  );
}
