import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { AuthProvider } from "../auth";
import type { Recording, Role, Transcript } from "../types";
import Login from "./Login";
import Recordings from "./Recordings";
import TranscriptPage from "./TranscriptPage";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  listRecordings: vi.fn(),
  getRecording: vi.fn(),
  getTranscript: vi.fn(),
  fetchAudio: vi.fn(),
  login: vi.fn(),
}));
import * as api from "../api";

const rec = (over: Partial<Recording> = {}): Recording => ({
  id: "rec_1",
  original_filename: "visit.wav",
  source: "upload",
  reference: null,
  sha256: "x".repeat(64),
  duration_sec: 65,
  status: "completed",
  stage: null,
  error: null,
  attempts: 0,
  options: {},
  created_at: "2026-11-26T10:00:00Z",
  completed_at: "2026-11-26T10:05:00Z",
  ...over,
});

function signedIn(role: Role) {
  sessionStorage.setItem("stt.session", JSON.stringify({ token: "t", role, username: "u" }));
}

function renderAt(path: string, ui: ReactNode, route = path) {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path={route} element={ui} />
          <Route path="/" element={<p>HOME</p>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>,
  );
}

beforeEach(() => {
  vi.mocked(api.fetchAudio).mockResolvedValue(new Blob(["x"], { type: "audio/wav" }));
  URL.createObjectURL = vi.fn(() => "blob:fake");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("Login", () => {
  it("signs in and redirects", async () => {
    vi.mocked(api.login).mockResolvedValue({ token: "t", role: "admin", username: "root" });
    renderAt("/login", <Login />);
    await userEvent.type(screen.getByLabelText("Username"), "root");
    await userEvent.type(screen.getByLabelText("Password"), "secret-password");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("HOME")).toBeInTheDocument();
    expect(api.loadSession()?.role).toBe("admin");
  });

  it("shows the server error", async () => {
    vi.mocked(api.login).mockRejectedValue(new api.ApiError(401, "Invalid username or password."));
    renderAt("/login", <Login />);
    await userEvent.type(screen.getByLabelText("Username"), "root");
    await userEvent.type(screen.getByLabelText("Password"), "wrong");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid username or password.");
  });
});

describe("Recordings list", () => {
  it("shows rows with status and stage", async () => {
    signedIn("viewer");
    vi.mocked(api.listRecordings).mockResolvedValue([
      rec(),
      rec({ id: "rec_2", original_filename: "live.webm", status: "processing", stage: "transcribe", duration_sec: null }),
      rec({ id: "rec_3", original_filename: "bad.wav", status: "failed", error: "boom" }),
    ]);
    renderAt("/rec", <Recordings />, "/rec");
    expect(await screen.findByText("visit.wav")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.getByText("Transcribing…")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toHaveAttribute("title", "boom");
    expect(screen.getAllByText("1:05").length).toBeGreaterThan(0);
  });

  it("shows an empty state", async () => {
    signedIn("viewer");
    vi.mocked(api.listRecordings).mockResolvedValue([]);
    renderAt("/rec", <Recordings />, "/rec");
    expect(await screen.findByText("No recordings yet.")).toBeInTheDocument();
  });
});

describe("Transcript page", () => {
  const original: Transcript = {
    recording_id: "rec_1",
    language: "en",
    speaker_count: 2,
    segments: [
      {
        speaker: "Speaker 1",
        start: 0,
        end: 2,
        confidence: 0.9,
        text: "Hello doctor.",
        words: [
          { word: "Hello", start: 0, end: 1, prob: 1 },
          { word: "doctor.", start: 1, end: 2, prob: 1 },
        ],
      },
      { speaker: "Speaker 2", start: 3, end: 5, confidence: 0.9, text: "Hi there." },
    ],
  };
  const redacted: Transcript = {
    recording_id: "rec_1",
    language: "en",
    speaker_count: 2,
    redaction: "pending",
    segments: [{ speaker: "Speaker 1", start: 0, end: 2, confidence: 0.9, text_redacted: null }],
  };

  it("admin sees original text, requests it, and gets the audio player", async () => {
    signedIn("admin");
    vi.mocked(api.getRecording).mockResolvedValue(rec());
    vi.mocked(api.getTranscript).mockResolvedValue(original);
    renderAt("/recordings/rec_1", <TranscriptPage />, "/recordings/:id");
    expect(await screen.findByText(/doctor\./)).toBeInTheDocument();
    expect(api.getTranscript).toHaveBeenCalledWith("rec_1", true);
    expect(screen.getAllByTestId("segment")).toHaveLength(2);
    expect(screen.getByText("Speaker 2")).toBeInTheDocument();
    await waitFor(() => expect(document.querySelector("audio")).not.toBeNull());
  });

  it("non-admin never requests original text or audio and sees the withheld notice", async () => {
    signedIn("viewer");
    vi.mocked(api.getRecording).mockResolvedValue(rec());
    vi.mocked(api.getTranscript).mockResolvedValue(redacted);
    renderAt("/recordings/rec_1", <TranscriptPage />, "/recordings/:id");
    expect(await screen.findByText(/redacted transcript is not available yet/i)).toBeInTheDocument();
    expect(api.getTranscript).toHaveBeenCalledWith("rec_1", false);
    expect(api.fetchAudio).not.toHaveBeenCalled();
    expect(document.querySelector("audio")).toBeNull();
    expect(screen.getByText("[text withheld]")).toBeInTheDocument();
  });

  it("shows the error and a Retry button when processing failed", async () => {
    signedIn("admin");
    vi.mocked(api.getRecording).mockResolvedValue(rec({ status: "failed", error: "GPU out of memory" }));
    renderAt("/recordings/rec_1", <TranscriptPage />, "/recordings/:id");
    expect((await screen.findAllByText("GPU out of memory")).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(api.getTranscript).not.toHaveBeenCalled();
  });

  it("hides Retry for recordings that never passed the gate", async () => {
    signedIn("admin");
    vi.mocked(api.getRecording).mockResolvedValue(rec({ status: "failed", sha256: null, error: "Not audio" }));
    renderAt("/recordings/rec_1", <TranscriptPage />, "/recordings/:id");
    expect((await screen.findAllByText("Not audio")).length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });
});
