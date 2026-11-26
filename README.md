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
