import { useEffect, useRef } from "react";
import { formatTimestamp } from "../lib/format";
import { findActiveSegment, findActiveWord, speakerSlot } from "../lib/transcript";
import type { Segment } from "../types";

interface Props {
  segments: Segment[];
  /** Original text is shown when present; otherwise the redacted text. */
  currentTime?: number;
  playing?: boolean;
  onSeek?: (sec: number) => void;
}

function segmentText(seg: Segment): string | null {
  return seg.text ?? seg.text_redacted ?? null;
}

/** Word-level highlighting is only exact when the shown text is the words themselves. */
function wordsMatchText(seg: Segment): boolean {
  return !!seg.words?.length && seg.text !== undefined && seg.text_raw === undefined;
}

export default function TranscriptView({ segments, currentTime, playing, onSeek }: Props) {
  const active = currentTime === undefined ? -1 : findActiveSegment(segments, currentTime);
  const refs = useRef<(HTMLDivElement | null)[]>([]);

  // Keep the spoken line in view during playback.
  useEffect(() => {
    if (playing && active >= 0) refs.current[active]?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [active, playing]);

  return (
    <div className="transcript">
      {segments.map((seg, i) => {
        const text = segmentText(seg);
        const isActive = i === active;
        const wordIdx = isActive && wordsMatchText(seg) ? findActiveWord(seg.words!, currentTime!) : -1;
        return (
          <div
            key={i}
            ref={(el) => {
              refs.current[i] = el;
            }}
            className={`seg speaker-${speakerSlot(seg.speaker)}${isActive ? " active" : ""}`}
            data-testid="segment"
          >
            <div className="seg-head">
              <span className="speaker">{seg.speaker}</span>
              {onSeek ? (
                <button type="button" className="link ts" onClick={() => onSeek(seg.start)} title="Play from here">
                  {formatTimestamp(seg.start)}
                </button>
              ) : (
                <span className="ts muted">{formatTimestamp(seg.start)}</span>
              )}
            </div>
            <p>
              {text === null ? (
                <span className="muted">[text withheld]</span>
              ) : wordsMatchText(seg) ? (
                seg.words!.map((w, j) => (
                  <span key={j} className={j === wordIdx ? "word on" : "word"}>
                    {w.word}{" "}
                  </span>
                ))
              ) : (
                text
              )}
            </p>
          </div>
        );
      })}
    </div>
  );
}
