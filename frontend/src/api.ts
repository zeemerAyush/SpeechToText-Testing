import type { Recording, RecordingOptions, Role, Session, Transcript } from "./types";

// The token lives in sessionStorage (cleared when the tab closes), never in localStorage.
const KEY = "stt.session";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function saveSession(s: Session): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage blocked: session lasts until reload */
  }
}

export function clearSession(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(fn: (() => void) | null): void {
  onUnauthorized = fn;
}

function authHeader(): Record<string, string> {
  const s = loadSession();
  return s ? { Authorization: `Bearer ${s.token}` } : {};
}

/** Turn a FastAPI error body into a readable message. */
export function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return "Some of the submitted values are invalid.";
  return `Request failed (${status}).`;
}

async function readBody(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, { ...init, headers: { ...authHeader(), ...(init.headers ?? {}) } });
  return handle<T>(res, (b) => b as T);
}

async function handle<T>(res: Response, parse: (body: unknown) => T): Promise<T> {
  if (res.status === 401 && loadSession()) {
    clearSession();
    onUnauthorized?.();
  }
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, await readBody(res)));
  return parse(await readBody(res));
}

export async function login(username: string, password: string): Promise<Session> {
  const res = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const body = (await readBody(res)) as { access_token?: string; role?: Role } | null;
  if (!res.ok || !body?.access_token || !body.role) {
    throw new ApiError(res.status, errorMessage(res.status, body));
  }
  return { token: body.access_token, role: body.role, username };
}

export const listRecordings = () => request<Recording[]>("/recordings");
export const getRecording = (id: string) => request<Recording>(`/recordings/${encodeURIComponent(id)}`);
export const retryRecording = (id: string) =>
  request<Recording>(`/recordings/${encodeURIComponent(id)}/retry`, { method: "POST" });
export const getTranscript = (id: string, original: boolean) =>
  request<Transcript>(`/recordings/${encodeURIComponent(id)}/transcript${original ? "?original=true" : ""}`);

async function fetchBlob(path: string): Promise<Blob> {
  const res = await fetch(path, { headers: authHeader() });
  if (res.status === 401 && loadSession()) {
    clearSession();
    onUnauthorized?.();
  }
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, await readBody(res)));
  return res.blob();
}

/** Audio is protected, so it is fetched with the token and played from a blob URL. */
export const fetchAudio = (id: string) => fetchBlob(`/recordings/${encodeURIComponent(id)}/audio`);

export async function downloadTranscriptJson(id: string, original: boolean): Promise<void> {
  const blob = await fetchBlob(
    `/recordings/${encodeURIComponent(id)}/export.json${original ? "?original=true" : ""}`,
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${id}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export interface UploadInput {
  file: File;
  source: "upload" | "browser_recording";
  reference: string;
  options: RecordingOptions;
  onProgress?: (fraction: number) => void;
}

/** XMLHttpRequest is used (not fetch) so the browser can report upload progress. */
export function uploadRecording(input: UploadInput): Promise<Recording> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", input.file, input.file.name);
    form.append("source", input.source);
    if (input.reference.trim()) form.append("reference", input.reference.trim());
    if (input.options.min_speakers) form.append("min_speakers", String(input.options.min_speakers));
    if (input.options.max_speakers) form.append("max_speakers", String(input.options.max_speakers));
    form.append("denoise", String(input.options.denoise));
    form.append("filler_removal", String(input.options.filler_removal));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/recordings");
    const token = loadSession()?.token;
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) input.onProgress?.(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error: could not reach the server."));
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error body */
      }
      if (xhr.status === 401 && loadSession()) {
        clearSession();
        onUnauthorized?.();
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as Recording);
      else reject(new ApiError(xhr.status, errorMessage(xhr.status, body)));
    };
    xhr.send(form);
  });
}
