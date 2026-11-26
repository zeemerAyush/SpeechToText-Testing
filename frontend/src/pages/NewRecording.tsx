import { useCallback, useRef, useState, type DragEvent, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { uploadRecording } from "../api";
import Recorder from "../components/Recorder";
import { formatBytes } from "../lib/format";
import { ACCEPTED_EXTENSIONS, hasAcceptedExtension } from "../lib/mime";

type Tab = "upload" | "record";

export default function NewRecording() {
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>("upload");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [speakers, setSpeakers] = useState("auto");
  const [denoise, setDenoise] = useState(true);
  const [fillerRemoval, setFillerRemoval] = useState(false);
  const [reference, setReference] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const file = tab === "upload" ? uploadFile : recordedFile;
  const busy = progress !== null;
  const onRecorded = useCallback((f: File | null) => setRecordedFile(f), []);

  function pick(f: File | undefined) {
    setError(null);
    if (!f) return;
    if (!hasAcceptedExtension(f.name)) {
      setUploadFile(null);
      setError(`Unsupported file type. Use one of: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return;
    }
    setUploadFile(f);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files[0]);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setProgress(0);
    const n = speakers === "auto" ? null : parseInt(speakers, 10);
    try {
      const rec = await uploadRecording({
        file,
        source: tab === "record" ? "browser_recording" : "upload",
        reference,
        options: { denoise, filler_removal: fillerRemoval, min_speakers: n, max_speakers: n },
        onProgress: setProgress,
      });
      navigate(`/recordings/${rec.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
      setProgress(null);
    }
  }

  return (
    <>
      <h1>New recording</h1>
      <form className="card" onSubmit={onSubmit}>
        <div className="tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === "upload"} onClick={() => setTab("upload")} disabled={busy}>
            Upload
          </button>
          <button type="button" role="tab" aria-selected={tab === "record"} onClick={() => setTab("record")} disabled={busy}>
            Record
          </button>
        </div>

        {tab === "upload" ? (
          <div
            className={`dropzone${dragging ? " dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            {uploadFile ? (
              <p>
                <strong>{uploadFile.name}</strong> <span className="muted">({formatBytes(uploadFile.size)})</span>
              </p>
            ) : (
              <p className="muted">Drag an audio file here</p>
            )}
            <button type="button" onClick={() => input.current?.click()} disabled={busy}>
              {uploadFile ? "Choose a different file" : "Choose file"}
            </button>
            <input
              ref={input}
              type="file"
              hidden
              accept={ACCEPTED_EXTENSIONS.join(",")}
              onChange={(e) => pick(e.target.files?.[0])}
              aria-label="Audio file"
            />
          </div>
        ) : (
          <Recorder onFile={onRecorded} />
        )}

        <fieldset className="options" disabled={busy}>
          <legend>Options</legend>
          <label>
            Expected number of speakers
            <select value={speakers} onChange={(e) => setSpeakers(e.target.value)}>
              <option value="auto">Auto-detect</option>
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
            Remove background noise
          </label>
          <label className="check">
            <input type="checkbox" checked={fillerRemoval} onChange={(e) => setFillerRemoval(e.target.checked)} />
            Remove filler words (um, uh)
          </label>
          <label>
            Reference (optional)
            <input
              value={reference}
              onChange={(e) => setReference(e.target.value)}
              maxLength={120}
              placeholder="e.g. clinic room 3, test session"
            />
            <span className="muted small">Do not enter patient names or other identifiers.</span>
          </label>
        </fieldset>

        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        {busy && (
          <div className="progress" role="progressbar" aria-valuenow={Math.round((progress ?? 0) * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
          </div>
        )}
        <button className="primary" type="submit" disabled={!file || busy}>
          {busy ? (progress !== null && progress >= 1 ? "Securing recording…" : "Uploading…") : "Upload & transcribe"}
        </button>
      </form>
    </>
  );
}
