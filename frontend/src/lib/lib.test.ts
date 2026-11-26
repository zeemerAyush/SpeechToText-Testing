import type { Segment } from "../types";
import { formatBytes, formatDuration, formatTimestamp } from "./format";
import { hasAcceptedExtension, pickRecorderFormat } from "./mime";
import { findActiveSegment, findActiveWord, isInFlight, speakerSlot, stageLabel } from "./transcript";

const seg = (start: number, end: number): Segment => ({ speaker: "Speaker 1", start, end, confidence: 1 });

describe("format", () => {
  it("formats durations", () => {
    expect(formatDuration(null)).toBe("-");
    expect(formatDuration(-1)).toBe("-");
    expect(formatDuration(0)).toBe("0:00");
    expect(formatDuration(65.4)).toBe("1:05");
    expect(formatDuration(3725)).toBe("1:02:05");
  });
  it("formats timestamps and sizes", () => {
    expect(formatTimestamp(42.9)).toBe("0:42");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("speakerSlot", () => {
  it("maps Speaker N to a stable palette slot that wraps", () => {
    expect(speakerSlot("Speaker 1")).toBe(0);
    expect(speakerSlot("Speaker 2")).toBe(1);
    expect(speakerSlot("Speaker 9")).toBe(0);
    expect(speakerSlot("unknown")).toBe(0);
  });
});

describe("findActiveSegment", () => {
  const segs = [seg(0, 2), seg(2.5, 4), seg(10, 12)];
  it("finds the segment containing t", () => {
    expect(findActiveSegment(segs, 0)).toBe(0);
    expect(findActiveSegment(segs, 1.9)).toBe(0);
    expect(findActiveSegment(segs, 3)).toBe(1);
    expect(findActiveSegment(segs, 11)).toBe(2);
  });
  it("returns -1 in gaps, before the start and after the end", () => {
    expect(findActiveSegment(segs, 2.2)).toBe(-1);
    expect(findActiveSegment(segs, 6)).toBe(-1);
    expect(findActiveSegment(segs, 12)).toBe(-1);
    expect(findActiveSegment([], 1)).toBe(-1);
    expect(findActiveSegment([seg(5, 6)], 1)).toBe(-1);
  });
});

describe("findActiveWord", () => {
  const words = [0, 0.5, 1.0].map((start) => ({ word: "w", start, end: start + 0.4, prob: 1 }));
  it("returns the last word that has started", () => {
    expect(findActiveWord(words, 0.1)).toBe(0);
    expect(findActiveWord(words, 0.6)).toBe(1);
    expect(findActiveWord(words, 5)).toBe(2);
    expect(findActiveWord(words, -1)).toBe(-1);
  });
});

describe("status helpers", () => {
  it("knows which statuses are still in flight", () => {
    expect(isInFlight("queued")).toBe(true);
    expect(isInFlight("processing")).toBe(true);
    expect(isInFlight("completed")).toBe(false);
    expect(isInFlight("failed")).toBe(false);
  });
  it("labels pipeline stages", () => {
    expect(stageLabel("transcribe")).toBe("Transcribing");
    expect(stageLabel(null)).toBe("");
    expect(stageLabel("novel")).toBe("novel");
  });
});

describe("mime", () => {
  it("prefers webm/opus, then falls back", () => {
    expect(pickRecorderFormat(() => true)).toEqual({ mime: "audio/webm;codecs=opus", ext: "webm" });
    expect(pickRecorderFormat((m) => m === "audio/mp4")).toEqual({ mime: "audio/mp4", ext: "m4a" });
    expect(pickRecorderFormat(() => false)).toBeNull();
  });
  it("checks upload extensions", () => {
    expect(hasAcceptedExtension("Visit.WAV")).toBe(true);
    expect(hasAcceptedExtension("notes.pdf")).toBe(false);
    expect(hasAcceptedExtension("noextension")).toBe(false);
  });
});
