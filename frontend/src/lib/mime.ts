/** Container formats the server accepts, in order of preference for browser recordings. */
const CANDIDATES: { mime: string; ext: string }[] = [
  { mime: "audio/webm;codecs=opus", ext: "webm" },
  { mime: "audio/webm", ext: "webm" },
  { mime: "audio/ogg;codecs=opus", ext: "ogg" },
  { mime: "audio/mp4", ext: "m4a" },
];

export interface RecorderFormat {
  mime: string;
  ext: string;
}

/** Pick the first recording format the browser supports, or null if none match. */
export function pickRecorderFormat(isSupported: (mime: string) => boolean): RecorderFormat | null {
  return CANDIDATES.find((c) => isSupported(c.mime)) ?? null;
}

export const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm"];

export function hasAcceptedExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}
