# GPU server setup (Linux)

Steps to run the app on the GPU server, with MongoDB as the database. Written for Ubuntu 22.04; adjust package
commands for other distributions. Do the steps in order: step 4 checks the code against a real MongoDB *before*
database authentication is switched on in step 5.

> Not in the git repository, so copy them over yourself if you want them there: `DESIGN.md` (git-ignored),
> `.env` (secrets, created on the server in step 6), and anything under `models/` or `storage/`.

## 1. Prerequisites

```bash
nvidia-smi                      # must list the GPU and driver
sudo apt update
sudo apt install -y git ffmpeg curl gnupg      # ffmpeg also provides ffprobe
python3.11 --version            # 3.11 is what the code was tested with
```

Ubuntu 22.04 has no `python3.11` by default: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update &&
sudo apt install -y python3.11 python3.11-venv`.

## 2. Get the code

```bash
git clone https://github.com/zeemerAyush/SpeechToText-Testing.git
cd SpeechToText-Testing
```

(If the repository is private, use a personal access token or an SSH deploy key.) To update later: `git pull`.

## 3. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# PyTorch with CUDA first (it must come from the CUDA index):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"     # must print True
pip install -r requirements.txt
python -m spacy download en_core_web_lg                         # once, for PHI redaction
```

## 4. MongoDB

Install (MongoDB 7.0 on Ubuntu 22.04; see the MongoDB docs for other releases):

```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt update && sudo apt install -y mongodb-org
sudo systemctl enable --now mongod
```

Check that it runs and listens on localhost only:

```bash
mongosh --quiet --eval 'db.runCommand({ping: 1})'      # { ok: 1 }
sudo ss -ltnp | grep 27017                               # must show 127.0.0.1:27017, not 0.0.0.0
```

Now run the whole backend test suite against the real database. Each test uses a temporary database that is dropped
afterwards. Expect `142 passed` (the same tests pass against the in-memory stand-in used during development):

```bash
TEST_MONGO_URI=mongodb://127.0.0.1:27017 pytest -q
```

## 5. Protect the database (strongly recommended)

Until a master key exists (see DESIGN.md), each recording's decryption key is stored in MongoDB, so anyone who can
read the database can decrypt the audio. Create an application user and turn authentication on.

```bash
mongosh
```
```js
use speech_to_text
db.createUser({ user: "stt_app", pwd: passwordPrompt(), roles: [{ role: "readWrite", db: "speech_to_text" }] })
exit
```

Edit `/etc/mongod.conf` so it contains:

```yaml
security:
  authorization: enabled
```

then `sudo systemctl restart mongod`. The connection string for the app is (URL-encode special characters in the
password, for example `@` as `%40`):

```
mongodb://stt_app:<password>@127.0.0.1:27017/speech_to_text?authSource=speech_to_text
```

## 6. Configuration

```bash
cp .env.example .env
chmod 600 .env
python -c "import secrets; print(secrets.token_urlsafe(48))"      # paste as JWT_SECRET
```

Edit `.env`: set `MONGO_URI` (from step 5), `JWT_SECRET`, and `REQUIRE_FFPROBE=true`. Do not reuse the `.env` from the
development PC.

## 7. Initialise the database and create users

```bash
python scripts/init_db.py                       # connects, creates indexes, prints a summary
python scripts/create_user.py admin admin       # asks for a password (10+ characters)
python scripts/create_user.py nurse uploader
python scripts/create_user.py reviewer viewer
```

`init_db.py` is safe to run again at any time. It reports a clear error if MongoDB is down or the credentials are wrong.

## 8. Web UI

`frontend/dist` is not in git. Either build it on the server (needs Node.js 20+; Ubuntu's own `nodejs` package is
older, so use NodeSource or nvm):

```bash
cd frontend && npm ci && npm run build && cd ..
```

or build it on your development PC and copy the `frontend/dist` folder to the same place on the server.

## 9. Models (once, with internet access)

The worker runs offline, so the models must be downloaded beforehand:

```bash
# Whisper (faster-whisper) into the Hugging Face cache:
HF_HUB_OFFLINE=0 python -c "from faster_whisper import download_model; download_model('large-v3')"
```

The pyannote speaker-diarization pipeline needs a Hugging Face account, an access token, and accepting the licences
of `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM`.
Download the three repositories into `models/pyannote/`, write a `config.yaml` there that points at the local files,
and set `PYANNOTE_CONFIG` to it (default `models/pyannote/config.yaml`). DeepFilterNet and Silero fetch their small
models on first use, so run one test file through the CLI while online (step 10) to let them download. The pyannote
and DeepFilterNet steps have not been exercised yet; expect to adjust them on the first real run.

## 10. Try the pipeline on one file

```bash
DEVICE=cuda WHISPER_MODEL=large-v3 python -m worker.cli sample.wav -o out.json
```

## 11. Run the app

In one terminal (or under systemd, see below), the worker:

```bash
DEVICE=cuda WHISPER_MODEL=large-v3 python -m worker.run_worker
```

In another, the API (it also serves the web UI):

```bash
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Check it: `curl http://127.0.0.1:8000/health` should show `"status":"ok"`, `"mongo":"ok"` and `"ffprobe":true`.
To use it from your own PC without exposing a port, open an SSH tunnel and browse to `http://127.0.0.1:8000`
(localhost counts as a secure page, so the microphone works):

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<server>
```

Smoke test in the browser: sign in as `admin`, upload a short WAV, and watch it go through the processing stages to
Completed. Then sign in as `reviewer` and confirm the transcript shows `[TAGS]` instead of names and numbers.

## Keeping it running

Run both commands as services (systemd units with `WorkingDirectory` set to the project, `ExecStart` pointing at
`.venv/bin/python -m worker.run_worker` and `.venv/bin/uvicorn ...`, and `Restart=on-failure`). The worker requeues a
recording that was interrupted by a restart, and temp files from a crashed job are wiped on startup.

## Backups

Recordings are kept forever. Back up `storage/audio/` and the database regularly (`mongodump --uri "<MONGO_URI>"`).
Without a master key, the database dump contains the audio decryption keys, so protect both backups as sensitive data.
