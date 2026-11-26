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

On the GPU server set `REQUIRE_FFPROBE=true` and install FFmpeg so uploads are fully decoded.

## Phase 2: command-line pipeline

```powershell
python -m worker.cli recording.wav -o transcript.json --device cuda --model large-v3
# options: --no-denoise --no-normalize --filler-removal --min-speakers 2 --max-speakers 4 --initial-prompt "..."
```

Needs the ML packages (see `requirements.txt`, Phase 2 section), FFmpeg, and a pyannote pipeline
downloaded once and referenced by `PYANNOTE_CONFIG` (default `models/pyannote/config.yaml`). Models run in
offline mode. The CLI reads plain audio directly (it does not go through the gate): use approved test audio only.
`pytest` covers the merge/post-process logic and the orchestration with fake stages; it needs none of the ML packages.
