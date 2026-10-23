# Training and research

Use Python 3.11 in a **separate** virtual environment:

```bash
python -m venv .venv-training
# Activate .venv-training using the command for your operating system.
python -m pip install -r training/requirements.txt
python -m jupyterlab notebooks/asl_mediapipe_final.ipynb
```

The first notebook cell locates the repository from the current directory.
The supplied landmarks CSV is `data/asl_mediapipe_keypoints_dataset.csv`.
To train from those landmarks, run the setup/import cells, skip image
extraction, and continue from the dataset-loading cell.

For image extraction, set `ASL_TRAINING_IMAGES` to your local image dataset
folder, whose subfolders are class labels. The default is
`data/raw/asl_alphabet_train/`. The original image dataset is not included.
The extracted CSV is written into `data/`.

Model saves target `models/asl_mediapipe_mlp_model.h5` and overwrite the
bundled model when those cells execute. Work on a Git branch and retain a
copy of a known working model before training. Check that the encoder label
order is exactly A–Z, `del`, `space` before deploying a replacement.

The notebook is a preserved research workflow, including an optional Tkinter
local webcam demo; those desktop cells need a GUI and a local camera and
are not supported by Community Cloud. They use pyttsx3 and may require
additional native speech/Tk packages for your operating system. Do not run
all notebook cells as a deployment task. Saved cell outputs were cleared to
keep the repository lean; results remain in earlier Git history.

The training environment uses TensorFlow/Keras and older compatible JAX
versions. It is intentionally independent of the web app requirements.
After changing the model, rerun the application tests, compare NumPy and
TensorFlow probabilities, and test live recognition before deployment.
