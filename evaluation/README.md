# Evaluation: measure quality and tune settings

Scores the pipeline against hand-corrected reference transcripts and compares settings side by side, so choices such
as "denoise on or off" and "which Whisper model" rest on numbers instead of impressions.

| Metric | Library | Meaning |
|---|---|---|
| **WER** (word error rate) | jiwer | Share of reference words that are wrong, missing or extra. Lower is better. |
| **DER** (diarization error rate) | pyannote.metrics | Share of speech time with the wrong speaker, missed, or falsely detected. Lower is better. |
| **Term recall** | built in | Of the medical terms spoken in the reference, how many the transcript got right. |
| **Speaker count right** | built in | Fraction of recordings where the number of speakers matched. |
| **RTF** | built in | Processing time / audio time (below 1 = faster than real time). |

Runs on the GPU machine with the real pipeline (needs the models from SERVER_SETUP.md). The tests use fake stages, so
the metric maths, reports and CLI are checked without any models.

## 1. Prepare the data

Use a handful of **approved test recordings** that look like real use: different rooms, noise levels and voices.
Aim for **at least 5 recordings and 10 minutes of audio**; the report warns you below that, because small differences
between settings are then just noise. Keep the folder outside git (`eval_data/` is git-ignored): it holds patient speech.

```
eval_data/
  visit01.wav
  visit01.ref.json     # text + speakers + times  -> WER and DER
  visit02.mp3
  visit02.ref.txt      # plain text only          -> WER only
  terms.txt            # optional: medical vocabulary, one term per line (# for comments)
```

**Making a reference.** Typing a transcript from scratch is slow, so start from a draft and correct it while listening:

```bash
python -m worker.cli eval_data/visit01.wav -o eval_data/visit01.ref.json --no-denoise
```

Then edit `visit01.ref.json` by hand: fix every wrong word, merge or split segments, fix speaker labels, and fix the
`start`/`end` times where a speaker change is misplaced. Only `speaker`, `start`, `end` and `text` of each segment are
read (extra fields are ignored). Correct it carefully: a draft that is only skimmed keeps the system's own mistakes and
makes the system look better than it is. Write numbers the way the pipeline does (digits) or accept small mismatches.

Wrong speaker *names* do not matter ("A"/"B" versus "Speaker 1/2"); DER finds the best matching itself.

## 2. Choose what to compare

`variants.json` is a list of settings sets. **The first is the baseline** that the others are compared with.
See `variants.example.json`:

```json
[
  {"name": "baseline"},
  {"name": "no-denoise", "denoise": false},
  {"name": "medical-prompt", "initial_prompt": "Surgery clinic visit. Terms: laparoscopic, cholecystectomy, ..."},
  {"name": "two-speakers", "min_speakers": 2, "max_speakers": 2},
  {"name": "small-model", "whisper_model": "small"}
]
```

Options per variant: `denoise`, `normalize`, `filler_removal`, `min_speakers`, `max_speakers`, `initial_prompt`,
`whisper_model`. Without `--variants`, the run compares *baseline* against *no-denoise*, the comparison DESIGN.md asks for.

## 3. Run

```bash
DEVICE=cuda WHISPER_MODEL=large-v3 python -m evaluation \
    --data eval_data --variants evaluation/variants.example.json --terms eval_data/terms.txt \
    --out eval_runs/first
```

Each variant runs over every recording, so *variants x recordings* full pipeline runs: start small. Variants that use
the same Whisper model share one model load; a different model is loaded afterwards and freed.

Useful flags: `--collar 0.25` (DER forgiveness around boundaries, the usual value), `--count-fillers` (count um/uh as
words; ignored by default so a filler-removal variant is judged fairly), `--save-transcripts` (also write each
hypothesis transcript; sensitive).

## 4. Read the results

`eval_runs/first/` contains `report.md` (read this), `results.json` (everything, for further analysis) and `runs.csv`
(one row per variant and recording).

- **Read the "Read first" notes**: small sample, skipped files, failed runs.
- **Summary table**: scores pool all words (or seconds of speech) across recordings, so long recordings count more.
- **Paired comparison**: per recording, variant minus baseline. Look at *Better / Worse*: a change that helps on 4 of 5
  recordings is more trustworthy than one that wins big on a single file. This table counts every recording equally, so
  it can disagree with the pooled summary when recordings differ in length.
- **RTF**: the first file per model includes model loading, so the median is reported.

## 5. Turn results into settings

| Finding | Setting |
|---|---|
| Denoise lowers WER consistently | keep `DENOISE=1` (the default); otherwise set `DENOISE=0` |
| A larger model is clearly better and fast enough | `WHISPER_MODEL=large-v3` (or the size that wins) |
| A prompt raises term recall without hurting WER | put it in a file and set `INITIAL_PROMPT_FILE=/path/prompt.txt` (or `INITIAL_PROMPT`); Whisper only uses about the last 220 tokens |
| DER improves when the speaker count is fixed | tell uploaders to set *Expected number of speakers* for those recordings |

Re-run the evaluation after changing a setting to confirm the gain on the same data.

## Limits

- WER compares words in time order, ignoring who spoke. Overlapping speech on a single microphone is hard for both
  Whisper and the diarizer; expect those stretches to dominate the errors.
- DER needs `pyannote.metrics`, which is installed together with `pyannote.audio` on the GPU machine. Without it the run
  still completes and the report says DER was not computed.
- A reference can contain mistakes too. With few recordings, treat small differences as unproven.
