"""Deployment contracts: stable entry point, portable assets, valid configuration."""

import json

import pytest
import tomllib

from app import BASE_DIR as PROJECT_ROOT
from asl_model import ASLModel

MODEL_PATH = PROJECT_ROOT / "asl_mediapipe_mlp_model.h5"
REFERENCE_IMAGE_PATH = PROJECT_ROOT / "ASL.png"


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
        (PROJECT_ROOT / "training/asl_mediapipe_final.ipynb").read_text(encoding="utf-8")
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
        "README.md",
        "training/requirements.txt",
        "asl_mediapipe_mlp_model.h5",
        "ASL.png",
    ],
)
def test_deployment_files_present(relative_path):
    assert (PROJECT_ROOT / relative_path).is_file()


@pytest.mark.parametrize("content_type", ["image/png", "audio/mp3", "video/mp4", "text/plain"])
@pytest.mark.parametrize("accept_encoding", [b"gzip", b"identity"])
def test_streamlit_media_middleware_sends_headers_before_body(content_type, accept_encoding):
    import asyncio

    from streamlit.web.server.starlette.starlette_gzip_middleware import (
        SelectiveGZipMiddleware,
    )

    body = b"media-payload" * 100
    messages = []

    async def endpoint(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", content_type.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    middleware = SelectiveGZipMiddleware(endpoint, minimum_size=1)
    asyncio.run(
        middleware(
            {
                "type": "http",
                "path": "/media/test",
                "headers": [(b"accept-encoding", accept_encoding)],
            },
            receive,
            send,
        )
    )
    assert messages[0]["type"] == "http.response.start"
    assert sum(message["type"] == "http.response.start" for message in messages) == 1
    if content_type.startswith(("audio/", "video/")):
        assert messages[-1]["body"] == body
