export type Role = "admin" | "uploader" | "viewer";

export type Status = "uploaded" | "gate_passed" | "queued" | "processing" | "completed" | "failed";

export interface RecordingOptions {
  denoise: boolean;
  filler_removal: boolean;
  min_speakers: number | null;
  max_speakers: number | null;
}

export interface Recording {
  id: string;
  original_filename: string;
  source: "upload" | "browser_recording";
  reference: string | null;
  sha256: string | null;
  duration_sec: number | null;
  status: Status;
  stage: string | null;
  error: string | null;
  attempts: number;
  options: Partial<RecordingOptions>;
  created_at: string;
  completed_at: string | null;
}

export interface Word {
  word: string;
  start: number;
  end: number;
  prob: number | null;
}

export interface Segment {
  speaker: string;
  start: number;
  end: number;
  confidence: number | null;
  /** Original text: only present in the admin ("original") view. */
  text?: string;
  text_raw?: string;
  words?: Word[];
  /** Redacted text: null until PHI redaction exists. */
  text_redacted?: string | null;
}

export type RedactionState = "pending" | "available" | "failed";

export interface Transcript {
  recording_id: string;
  language: string;
  speaker_count: number;
  /** Redacted view: a state string. Original (admin) view: the stored status object. */
  redaction?: RedactionState | { status: string; engine?: string | null; error?: string };
  segments: Segment[];
  full_text?: string;
  full_text_redacted?: string | null;
}

export interface Session {
  token: string;
  role: Role;
  username: string;
}
