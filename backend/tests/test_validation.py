import pytest

from app.gate.validation import ValidationError, detect_type, validate_audio
from tests.conftest import make_wav


def run(tmp_path, name, data, **kw):
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    args = dict(max_bytes=1024 * 1024, max_duration_sec=30, require_ffprobe=False)
    args.update(kw)
    return validate_audio(p, name, **args)


def test_valid_wav(tmp_path):
    r = run(tmp_path, "a.wav", make_wav(1.0))
    assert r.container == "wav" and 0.9 < r.duration_sec < 1.1


def test_extension_must_match_signature(tmp_path):
    with pytest.raises(ValidationError, match="does not match"):
        run(tmp_path, "a.mp3", make_wav())


def test_non_audio_rejected(tmp_path):
    with pytest.raises(ValidationError, match="not a recognised"):
        run(tmp_path, "a.wav", b"MZ\x90\x00 this is an executable" * 10)


def test_unsupported_extension(tmp_path):
    with pytest.raises(ValidationError, match="Unsupported"):
        run(tmp_path, "a.exe", make_wav())


def test_empty_file(tmp_path):
    with pytest.raises(ValidationError, match="empty"):
        run(tmp_path, "a.wav", b"")


def test_oversize(tmp_path):
    with pytest.raises(ValidationError) as e:
        run(tmp_path, "a.wav", make_wav(1.0), max_bytes=100)
    assert e.value.status_code == 413


def test_too_long(tmp_path):
    with pytest.raises(ValidationError, match="duration"):
        run(tmp_path, "a.wav", make_wav(3.0), max_duration_sec=1)


def test_malformed_wav_with_valid_header(tmp_path):
    with pytest.raises(ValidationError):
        run(tmp_path, "a.wav", b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20)


def test_detect_type():
    assert detect_type(b"\x1a\x45\xdf\xa3" + b"\0" * 12) == "webm"
    assert detect_type(b"OggS" + b"\0" * 12) == "ogg"
    assert detect_type(b"\0\0\0\x20ftypM4A " + b"\0" * 4) == "m4a"
    assert detect_type(b"random garbage!!") is None
