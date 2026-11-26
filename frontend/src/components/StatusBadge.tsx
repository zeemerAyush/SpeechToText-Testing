import { STATUS_LABELS, isInFlight, stageLabel } from "../lib/transcript";
import type { Recording } from "../types";

export default function StatusBadge({ rec }: { rec: Pick<Recording, "status" | "stage" | "error"> }) {
  const stage = rec.status === "processing" ? stageLabel(rec.stage) : "";
  return (
    <span className="status-wrap">
      <span className={`badge badge-${rec.status}`} title={rec.error ?? undefined}>
        {isInFlight(rec.status) && <span className="pulse" aria-hidden="true" />}
        {STATUS_LABELS[rec.status]}
      </span>
      {stage && <span className="muted small">{stage}…</span>}
    </span>
  );
}
