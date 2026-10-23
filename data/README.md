# Landmark data

`asl_mediapipe_keypoints_dataset.csv` is the existing landmark dataset moved
from the repository root. Each row contains 63 hand coordinates and a label.
It is preserved for training and is not opened by the deployed app.

Raw training images are not included. Place local images under `data/raw/`
or set `ASL_TRAINING_IMAGES`; raw data is excluded from Git. Verify data
provenance and redistribution rights before sharing new datasets.

See [training instructions](../training/README.md).
