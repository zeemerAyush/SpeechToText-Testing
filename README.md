# Speech-to-Text (surgeon–patient transcription)

See [DESIGN.md](DESIGN.md). Currently implemented: **Phase 1, guardrail gate**.

## Dev setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest                      # runs against mongomock; no MongoDB or ffmpeg needed
```

## Running (needs MongoDB on 127.0.0.1)

```powershell
copy .env.example .env      # then edit
python scripts/create_user.py admin admin
uvicorn app.main:app --app-dir backend --port 8000
python -m worker.run_worker                                       # separate terminal
```

Set `REQUIRE_FFPROBE=true` (with FFmpeg installed) so uploads are fully decoded.

## Phase 2: command-line pipeline

```powershell
python -m worker.cli recording.wav -o transcript.json --device cuda --model large-v3
# options: --no-denoise --no-normalize --filler-removal --min-speakers 2 --max-speakers 4 --initial-prompt "..."
```

Needs the ML packages (see `requirements.txt`, Phase 2 section), FFmpeg, and a pyannote pipeline
downloaded once and referenced by `PYANNOTE_CONFIG` (default `models/pyannote/config.yaml`). Models run in
offline mode. The CLI reads plain audio directly (it does not go through the gate): use approved test audio only.
`pytest` covers the merge/post-process logic and the orchestration with fake stages; it needs none of the ML packages.

## Phase 3: API and job lifecycle

`uploaded -> gate_passed -> queued -> processing -> completed | failed`. The worker (`python -m worker.run_worker`)
runs the real pipeline, stores the transcript in `transcripts`, and records the current `stage` for progress.
Interrupted jobs are requeued when the worker restarts.

| Endpoint | Roles |
|---|---|
| `POST /recordings` (options: `min_speakers`, `max_speakers`, `denoise`, `filler_removal`) | admin, uploader |
| `GET /recordings`, `GET /recordings/{id}` | all |
| `GET /recordings/{id}/transcript`, `GET /recordings/{id}/export.json` | all (redacted view) |
| same with `?original=true` | admin |
| `GET /recordings/{id}/audio` (decrypted on the fly, supports Range) | admin |
| `POST /recordings/{id}/retry` (failed recordings only) | admin |

PHI redaction is Phase 5. Until then the redacted view carries no text (fail closed): only admins can read
transcripts, via `?original=true`.

## Phase 4: web UI

React + Vite + TypeScript in `frontend/`. FastAPI serves the built app at `/` (same origin, so no CORS) with
security headers. Page URLs use `#/...` so they never collide with API paths.

```powershell
cd frontend
npm ci
npm test              # 24 unit/component tests
npm run build         # type check + production build into frontend/dist
npm run dev           # hot reload; proxies the API to http://127.0.0.1:8000 (override with BACKEND_URL)
```

Try the whole app **without MongoDB, GPU or models** (development only, in-memory, fake pipeline):

```powershell
python scripts/dev_server.py --port 8010      # logins admin / uploader / viewer, password devpassword123
```

Pages: login, recordings list (auto-refreshes while jobs run), new recording (upload with drag-and-drop, or record
in the browser with mic selector, level meter, timer, pause/resume), and the transcript view (colour-coded speakers,
timestamps; admins also get the audio player with click-to-seek and live highlighting).

## Phase 5: PHI redaction

Every transcript segment gets a `text_redacted` copy with identifiers replaced by tags such as `[PERSON]`,
`[PHONE_NUMBER]`, `[EMAIL_ADDRESS]`, `[DATE_TIME]`, `[LOCATION]`, `[MEDICAL_RECORD_NUMBER]`, `[ID_NUMBER]`.
Detection has two layers (`worker/pipeline/redact.py`):

- **Rules** (pure Python, always on): spoken digit runs ("five five five ..."), record/insurance numbers after a cue
  ("MRN is ..."), emails, SSNs, long digit strings, street addresses and specific dates.
- **Presidio + spaCy** (names, places, plus Presidio's own recognizers). Needs `presidio-analyzer`, `spacy` and the
  `en_core_web_lg` model installed once (`python -m spacy download en_core_web_lg`); it then runs fully offline.

Tuned for clinical speech: relative times ("next week", "two days"), doses and vitals are kept, and eponymous
conditions ("Parkinson's", "Foley catheter") are not mistaken for people (`ALLOW_LIST`).

**Fail closed.** If redaction cannot run (for example the spaCy model is missing), the transcript is still stored for
admins, but `text_redacted` stays empty, so viewers and uploaders see no text and a notice explaining why. Admins can
switch between *Original* and *Redacted* on the transcript page to check the result. Re-run redaction later with:

```powershell
python -m worker.redact_backfill      # redacts stored transcripts that have no (or failed) redaction
```

Redaction is best effort (names in lowercase or noisy speech can be missed); access control and encryption remain the
primary protection. Audio itself is not redacted.
