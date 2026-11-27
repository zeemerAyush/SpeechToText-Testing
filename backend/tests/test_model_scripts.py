import importlib.util
import os
from pathlib import Path

import pytest

import worker.config as wconfig

ROOT = Path(__file__).resolve().parents[2]

PYANNOTE_CONFIG = """\
version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: pyannote/wespeaker-voxceleb-resnet34-LM
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: pyannote/segmentation-3.0
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_models = load_script("download_models")


# ---------------------------------------------------------------- pyannote offline config

def test_rewrite_points_only_the_two_checkpoint_lines_at_local_files():
    out = download_models.rewrite_pyannote_config(PYANNOTE_CONFIG, "/m/seg/pytorch_model.bin", "/m/emb/pytorch_model.bin")
    assert '    segmentation: "/m/seg/pytorch_model.bin"' in out
    assert '    embedding: "/m/emb/pytorch_model.bin"' in out
    assert "pyannote/segmentation-3.0" not in out and "wespeaker" not in out
    # everything else is untouched, including the nested "segmentation:" block and similarly named keys
    assert "  segmentation:\n    min_duration_off: 0.0" in out
    assert "embedding_batch_size: 32" in out and "segmentation_batch_size: 32" in out
    assert "threshold: 0.7045654963945799" in out
    assert len(out.splitlines()) == len(PYANNOTE_CONFIG.splitlines())


def test_rewritten_config_is_valid_yaml_with_the_new_paths():
    yaml = pytest.importorskip("yaml")
    out = download_models.rewrite_pyannote_config(PYANNOTE_CONFIG, "/m/seg.bin", "/m/emb.bin")
    params = yaml.safe_load(out)["pipeline"]["params"]
    assert params["segmentation"] == "/m/seg.bin" and params["embedding"] == "/m/emb.bin"
    assert yaml.safe_load(out)["params"]["segmentation"] == {"min_duration_off": 0.0}


@pytest.mark.parametrize(
    "broken",
    [
        PYANNOTE_CONFIG.replace("    segmentation: pyannote/segmentation-3.0\n", ""),   # key missing
        PYANNOTE_CONFIG + "    embedding: another/repo\n",                              # key duplicated
    ],
)
def test_rewrite_refuses_a_config_it_does_not_understand(broken):
    with pytest.raises(ValueError, match="Expected exactly one"):
        download_models.rewrite_pyannote_config(broken, "/a", "/b")


def test_download_script_with_every_step_skipped_prints_the_env_lines(capsys, tmp_path, monkeypatch):
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(var, "1")  # monkeypatch restores both after the script switches them off
    code = download_models.main(["--models-dir", str(tmp_path), "--skip", *download_models.STEPS])
    out = capsys.readouterr().out
    assert code == 0
    assert "WHISPER_MODEL_DIR=" in out and "PYANNOTE_CONFIG=" in out and "check_models.py" in out
    assert os.environ["HF_HUB_OFFLINE"] == "0"       # the downloader needs the network; the worker never does


def test_download_script_reports_a_failing_step_and_continues(capsys, tmp_path, monkeypatch):
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(var, "1")
    monkeypatch.setattr(download_models, "step_pyannote", lambda *a: (_ for _ in ()).throw(RuntimeError("gated")))
    monkeypatch.setattr(download_models, "step_whisper", lambda *a: "saved")
    code = download_models.main(["--models-dir", str(tmp_path), "--skip", "deepfilternet", "spacy", "silero"])
    captured = capsys.readouterr()
    assert code == 1
    assert "[OK]     whisper" in captured.out and "[FAILED] pyannote: RuntimeError: gated" in captured.err


# ---------------------------------------------------------------- worker config

def test_env_file_fills_missing_values_but_never_overrides():
    pytest.importorskip("dotenv")
    env = {"DEVICE": "cpu"}  # already set, e.g. exported in the shell
    path = ROOT / "backend" / "tests" / "_tmp_env_for_test"
    path.write_text('# comment\nDEVICE=cuda\nWHISPER_MODEL="large-v3"\nEMPTY=\nSPACY_MODEL=en_core_web_lg\n',
                    encoding="utf-8")
    try:
        wconfig.load_env_file(path, env)
    finally:
        path.unlink()
    assert env["DEVICE"] == "cpu"                   # the shell wins over the file
    assert env["WHISPER_MODEL"] == "large-v3" and env["SPACY_MODEL"] == "en_core_web_lg"
    assert env["EMPTY"] == ""


def test_missing_env_file_is_fine():
    env = {}
    wconfig.load_env_file(Path("/no/such/.env"), env)
    assert env == {}


def test_relative_model_paths_resolve_from_the_project_root():
    assert wconfig.resolve_path("models/pyannote/config.yaml") == str(ROOT / "models" / "pyannote" / "config.yaml")
    absolute = str(Path("/srv/models/x.yaml").resolve())
    assert wconfig.resolve_path(absolute) == absolute


def test_model_libraries_are_offline_and_telemetry_is_off_by_default():
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "0"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_check_script_is_importable_and_documents_itself():
    text = (ROOT / "scripts" / "check_models.py").read_text(encoding="utf-8")
    assert "OFFLINE" in text and "scripts/check_models.py" in text
