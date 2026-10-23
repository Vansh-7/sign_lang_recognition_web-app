"""Application limits and paths, independent of the current working directory."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "asl_mediapipe_mlp_model.h5"
REFERENCE_IMAGE_PATH = PROJECT_ROOT / "assets" / "asl_alphabet.png"
CLASSES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("del", "space")
CONFIDENCE_THRESHOLD = 0.8
MAX_SENTENCE_LENGTH = 2000
MAX_FRAME_WIDTH = 480
INFERENCE_INTERVAL = 0.1


def configure_runtime():
    """Keep native libraries from consuming every CPU on the shared host."""
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
