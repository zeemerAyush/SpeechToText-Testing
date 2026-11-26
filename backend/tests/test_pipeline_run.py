"""Orchestration tests with fake stages (no ML models, no FFmpeg)."""
import json
from array import array
from pathlib import Path

import pytest

from worker.pipeline import wavio
from worker.pipeline.run import Options, PipelineConfig, Stages, process_audio


def fake_stages(calls=None, fail_in=None):
    calls = calls if calls is not None else []

    def step(name, fn):
        def inner(*a):
            calls.append(name)
            if fail_in == name:
                raise RuntimeError(f"{name} exploded")
            return fn(*a)
        return inner

    def write_wav(src, dst, *_):
        wavio.write_pcm16(dst, array("h", [0] * 32000), 16000)  # 2 seconds

    words = [
        {"word": " Hello", "start": 0.1, "end": 0.5, "prob": 0.9},
        {"word": " doctor.", "start": 0.5, "end": 1.0, "prob": 0.9},
        {"word": " Hi,", "start": 1.2, "end": 1.5, "prob": 0.9},
        {"word": " um", "start": 1.5, "end": 1.7, "prob": 0.9},
        {"word": " come", "start": 1.7, "end": 1.9, "prob": 0.9},
    ]
    turns = [{"start": 0, "end": 1.1, "speaker": "SPEAKER_05"}, {"start": 1.1, "end": 2, "speaker": "SPEAKER_01"}]
    return Stages(
        convert=step("convert", write_wav),
        denoise=step("denoise", write_wav),
        vad=step("vad", lambda wav: [(0.0, 2.0)]),
        transcribe=step("transcribe", lambda wav, prompt: words),
        diarize=step("diarize", lambda wav, lo, hi: turns),
    ), calls


def leftover(root: Path):
    return [p for p in root.rglob("*") if p.is_file()]


def test_end_to_end_shape_and_speaker_order(tmp_path):
    stages, calls = fake_stages()
    out = process_audio(tmp_path / "in.wav", Options(), stages, PipelineConfig(), tmp_path / "tmp")
    assert calls == ["convert", "denoise", "vad", "transcribe", "diarize"]
    assert out["language"] == "en" and out["duration_sec"] == 2.0 and out["speaker_count"] == 2
    s1, s2 = out["segments"]
    assert (s1["speaker"], s1["text"]) == ("Speaker 1", "Hello doctor.")  # SPEAKER_05 spoke first
    assert (s2["speaker"], s2["text"]) == ("Speaker 2", "Hi, um come")
    assert s1["start"] == 0.1 and s1["end"] == 1.0
    assert s1["words"][0]["word"] == "Hello"
    assert out["full_text"] == "Speaker 1: Hello doctor.\nSpeaker 2: Hi, um come"
    assert set(out["pipeline"]["timings_sec"]) == {"convert", "denoise", "vad", "transcribe", "diarize"}
    json.dumps(out)  # must be JSON-serialisable


def test_denoise_is_skippable(tmp_path):
    stages, calls = fake_stages()
    process_audio(tmp_path / "in.wav", Options(denoise=False), stages, PipelineConfig(), tmp_path / "tmp")
    assert "denoise" not in calls


def test_filler_removal_option(tmp_path):
    stages, _ = fake_stages()
    out = process_audio(tmp_path / "in.wav", Options(filler_removal=True), stages, PipelineConfig(), tmp_path / "tmp")
    assert out["segments"][1]["text"] == "Hi, come"
    assert out["segments"][1]["text_raw"] == "Hi, um come"


def test_temp_files_wiped_on_success_and_failure(tmp_path):
    root = tmp_path / "tmp"
    stages, _ = fake_stages()
    process_audio(tmp_path / "in.wav", Options(), stages, PipelineConfig(), root)
    assert leftover(root) == []

    stages, _ = fake_stages(fail_in="transcribe")
    with pytest.raises(RuntimeError, match="transcribe exploded"):
        process_audio(tmp_path / "in.wav", Options(), stages, PipelineConfig(), root)
    assert leftover(root) == []
    assert list(root.iterdir()) == []  # the job directory itself is gone too


def test_on_stage_callback_reports_each_stage_in_order(tmp_path):
    stages, _ = fake_stages()
    seen = []
    process_audio(tmp_path / "in.wav", Options(), stages, PipelineConfig(), tmp_path / "tmp", on_stage=seen.append)
    assert seen == ["convert", "denoise", "vad", "transcribe", "diarize"]
