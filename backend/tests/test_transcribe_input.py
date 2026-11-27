import wave

import pytest

np = pytest.importorskip("numpy")

from worker.pipeline import transcribe  # noqa: E402


def write_wav(path, samples, rate=16000, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_pipeline_wav_is_decoded_to_float_samples(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, [0, 16384, -16384, 32767])
    out = transcribe.read_wav_16k_mono(path)
    assert out.dtype == np.float32
    assert out.tolist() == pytest.approx([0.0, 0.5, -0.5, 32767 / 32768])


@pytest.mark.parametrize("rate,channels", [(44100, 1), (16000, 2)])
def test_other_wav_formats_are_left_to_faster_whisper(tmp_path, rate, channels):
    path = tmp_path / "b.wav"
    write_wav(path, [0, 1, 2, 3], rate=rate, channels=channels)
    assert transcribe.read_wav_16k_mono(path) is None


def test_non_wav_file_is_left_to_faster_whisper(tmp_path):
    path = tmp_path / "c.wav"
    path.write_bytes(b"not a wav file")
    assert transcribe.read_wav_16k_mono(path) is None


def test_transcribe_passes_samples_not_a_path(tmp_path):
    path = tmp_path / "d.wav"
    write_wav(path, [0, 100, 200])
    seen = {}

    class FakeWord:
        word, start, end, probability = " hi", 0.0, 0.5, 0.9

    class FakeSegment:
        words = [FakeWord()]

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            return [FakeSegment()], None

    words = transcribe.transcribe(FakeModel(), path)
    assert isinstance(seen["audio"], np.ndarray)
    assert words == [{"word": " hi", "start": 0.0, "end": 0.5, "prob": 0.9}]
