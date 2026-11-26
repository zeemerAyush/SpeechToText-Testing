import type { Segment, Status, Word } from "../types";

const PALETTE_SIZE = 8;

/** Stable colour slot (0-7) for a label such as "Speaker 3". */
export function speakerSlot(label: string): number {
  const m = /(\d+)\s*$/.exec(label);
  const n = m ? parseInt(m[1], 10) : 1;
  return (Math.max(n, 1) - 1) % PALETTE_SIZE;
}

/** Index of the segment being spoken at time t, or -1 when t falls in a gap. */
export function findActiveSegment(segments: Segment[], t: number): number {
  // Segments are sorted by start; binary search for the last one starting at or before t.
  let lo = 0;
  let hi = segments.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segments[mid].start <= t) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (found === -1) return -1;
  return t < segments[found].end ? found : -1;
}

/** Index of the word being spoken at time t (last word starting at or before t, within the segment). */
export function findActiveWord(words: Word[], t: number): number {
  let found = -1;
  for (let i = 0; i < words.length; i++) {
    if (words[i].start <= t) found = i;
    else break;
  }
  return found;
}

export const IN_FLIGHT: Status[] = ["uploaded", "gate_passed", "queued", "processing"];

export function isInFlight(status: Status): boolean {
  return IN_FLIGHT.includes(status);
}

const STAGE_LABELS: Record<string, string> = {
  starting: "Starting",
  convert: "Converting audio",
  denoise: "Removing background noise",
  vad: "Detecting speech",
  transcribe: "Transcribing",
  diarize: "Separating speakers",
};

export function stageLabel(stage: string | null): string {
  return stage ? (STAGE_LABELS[stage] ?? stage) : "";
}

export const STATUS_LABELS: Record<Status, string> = {
  uploaded: "Uploaded",
  gate_passed: "Checked",
  queued: "Queued",
  processing: "Processing",
  completed: "Completed",
  failed: "Failed",
};
