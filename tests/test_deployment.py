"""Deployment contracts: stable entry point, portable assets, valid configuration."""

import json
import tomllib

import pytest

from asl_recognition.config import MODEL_PATH, PROJECT_ROOT, REFERENCE_IMAGE_PATH
from asl_recognition.model import ASLModel


def test_runtime_assets_resolve_outside_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert MODEL_PATH.is_absolute() and MODEL_PATH.is_file()
    assert REFERENCE_IMAGE_PATH.is_absolute() and REFERENCE_IMAGE_PATH.is_file()
    assert len(ASLModel(MODEL_PATH).layers) == 3


def test_streamlit_configuration_and_secret_template():
    config = tomllib.loads((PROJECT_ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    assert config["server"]["headless"] is True
    assert config["client"]["showErrorDetails"] == "none"
    secrets = tomllib.loads(
        (PROJECT_ROOT / ".streamlit/secrets.toml.example").read_text(encoding="utf-8")
    )
    assert secrets["twilio"]["account_sid"] == "YOUR_TWILIO_ACCOUNT_SID"
    assert not (PROJECT_ROOT / ".streamlit/secrets.toml").exists()


def test_notebook_paths_and_code_are_valid():
    notebook = json.loads(
        (PROJECT_ROOT / "notebooks/asl_mediapipe_final.ipynb").read_text(encoding="utf-8")
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"]
            compile("".join(cell["source"]), "<notebook cell>", "exec")
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "C:\\Users\\" not in source
    assert "ASL_TRAINING_IMAGES" in source


def test_runtime_requirements_exclude_training_frameworks():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = {
        line.split("==")[0].lower() for line in requirements if line and not line.startswith("#")
    }
    assert "tensorflow" not in packages
    assert "tensorflow-cpu" not in packages
    assert "keras" not in packages
    assert "opencv-python-headless" not in packages
    assert "opencv-contrib-python" in packages


@pytest.mark.parametrize(
    "relative_path",
    [
        "app.py",
        ".github/workflows/ci.yml",
        "docs/deployment.md",
        "models/asl_mediapipe_mlp_model.h5",
        "assets/asl_alphabet.png",
    ],
)
def test_deployment_files_present(relative_path):
    assert (PROJECT_ROOT / relative_path).is_file()
