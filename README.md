# 🖐️ Real-Time ASL Recognition

Recognize ASL alphabet hand poses from a browser webcam, build an editable sentence,
and turn the result into speech or a downloadable text file.

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![MediaPipe 0.10.14](https://img.shields.io/badge/MediaPipe-0.10.14-0097A7)

**🌐 [Open the live demo](https://asl-recognition-7.streamlit.app/)**

## Key Features

- **Webcam recognition:** Browser-based video streaming with a live sign and sentence overlay.
- **Stable predictions:** Adjustable hold and cooldown settings reduce accidental inputs.
- **Editable text:** Correct recognized text or clear it while keeping the camera running.
- **Text-to-speech:** Read the sentence aloud using gTTS.
- **Text export:** Download the current sentence as a `.txt` file.
- **ASL reference:** View the alphabet chart beside the webcam feed.

## Technology Stack

| Area | Technologies |
| --- | --- |
| Interface | Streamlit |
| Video streaming | Streamlit WebRTC, PyAV, Twilio TURN / public STUN fallback |
| Hand tracking | MediaPipe Hands, OpenCV |
| Model inference | NumPy, h5py |
| Speech | gTTS |
| Training only | TensorFlow/Keras, scikit-learn, Pandas, Matplotlib, JupyterLab |

Runtime and training dependencies are separate. The deployed app reads the existing
H5 weights without importing TensorFlow or Keras.

## How It Works

1. The browser sends webcam frames to the app through WebRTC.
2. OpenCV resizes and mirrors the frame; MediaPipe tracks one hand.
3. The 21 landmarks are flattened into 63 coordinates in x/y/z order.
4. The neural network predicts one of 28 classes: **A–Z**, **`del`**, or **`space`**.
5. Predictions above **0.8 confidence** must remain stable before updating the sentence.
6. Sentence tools let you edit, speak, download, or clear the result.

Capture is limited to 15 FPS, with recognition running at most 10 times per second.
Stale frames are dropped, and the sentence panel refreshes every 0.5 seconds without
rerunning the whole page. Each camera session closes its own tracker when stopped.

## Model Architecture and Inference

The saved model is a multilayer perceptron trained on landmark coordinates:

| Layer | Output | Activation / configuration |
| --- | --- | --- |
| Input | 63 features | 21 landmarks × 3 coordinates |
| Dense | 128 units | ReLU |
| Dropout | 128 units | Rate 0.2; training only |
| Dense | 64 units | ReLU |
| Dropout | 64 units | Rate 0.2; training only |
| Dense | 28 classes | Softmax |

```text
Webcam → hand landmarks → 63 coordinates → MLP probabilities
       → confidence / stability / cooldown checks → editable sentence
```

`asl_model.py` loads the dense-layer weights from
`models/asl_mediapipe_mlp_model.h5`. NumPy performs the forward pass; dropout is
inactive during inference. Model weights are cached across sessions. Sliders count
**recognition samples**, rather than captured camera frames. Text is capped at
2,000 characters, with the last 24 shown in the video overlay.

## Project Structure

```text
sign_lang_recognition_web-app/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── assets/
│   └── ASL.png
├── models/
│   └── asl_mediapipe_mlp_model.h5
├── training/
│   ├── asl_mediapipe_final.ipynb
│   ├── asl_mediapipe_keypoints_dataset.csv
│   └── requirements.txt
├── tests/
│   ├── conftest.py
│   ├── test_app.py
│   └── test_deployment.py
├── app.py
├── asl_model.py
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

`app.py` remains the Streamlit entry point. Model and image paths resolve relative
to that file, so they do not depend on the current working directory. The training
notebook and complete dataset are retained and are not loaded by the running app.

## Local Installation

Use **Python 3.11**, a webcam, and a browser with camera access.

```bash
git clone https://github.com/Vansh-7/sign_lang_recognition_web-app.git
cd sign_lang_recognition_web-app
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use these environment commands instead:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Debian/Ubuntu, install the OpenCV system libraries if needed:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## Run the Application

From the repository root, with the environment active:

```bash
python -m streamlit run app.py
```

Select **Start**, allow camera access, and hold one hand steady in good lighting.
Use **Sensitivity Settings** to adjust how long a sign must be held and the wait
between letters. Select **Stop** when finished.

File watching is disabled in the deployment configuration. For automatic local
reloads, run `python -m streamlit run app.py --server.fileWatcherType=auto`.

To run the automated checks:

```bash
python -m pip install pytest
python -m pytest -q
```

## Training and Dataset

The supplied CSV contains **59,803 samples**, each with **63 landmark features** and
one label from the 28 output classes. The notebook uses a label encoder, an 80/20
train/test split, and a 20% validation split within training. Its training cell
uses Adam, categorical cross-entropy, 10 epochs, and a batch size of 128.

Create a separate Python 3.11 environment for training so its TensorFlow/JAX
requirements do not mix with the web app's environment:

```bash
python3.11 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -r training/requirements.txt
python -m jupyterlab training/asl_mediapipe_final.ipynb
```

On Windows, create it with `py -3.11 -m venv .venv-training` and activate
`.\.venv-training\Scripts\Activate.ps1` before the install command.

- To use the supplied CSV, run the setup/import cells, skip image extraction,
  and continue from the dataset-loading cell.
- Raw training images are not included. To extract new landmarks, point
  `ASL_TRAINING_IMAGES` at an image dataset organized into class folders.
- The notebook resolves the CSV in `training/`, the model in `models/`, and
  the reference image in `assets/`, whether opened from the root or `training/`.
- Training/extraction cells overwrite the corresponding model/CSV when run;
  keep a backup and verify class order before replacing the deployed model.
- The optional Tkinter webcam demo requires a local GUI and camera. It does not
  run on Streamlit Cloud. Saved notebook outputs are omitted from Git.

## Streamlit Community Cloud Deployment

### Select Python 3.11 explicitly

`mediapipe==0.10.14` has a Python 3.11 Linux wheel, but no Python 3.14 wheel.
Keep the pinned MediaPipe version and select the compatible interpreter at deploy
time; replacing or deleting the model does not fix an interpreter mismatch.
See the [published MediaPipe files](https://pypi.org/project/mediapipe/0.10.14/#files).

1. In your [Community Cloud workspace](https://share.streamlit.io/), select
   **Create app**, then choose the existing GitHub repository.
2. Set the repository to `Vansh-7/sign_lang_recognition_web-app` and select the
   branch containing these changes (`main` after merge).
3. Set **Main file path** to **`app.py`** and the desired app subdomain to
   **`asl-recognition-7`**, if available.
4. Open **Advanced settings** and select **Python 3.11** in the Python dropdown.
5. Add any Twilio secrets, save the settings, and select **Deploy**.
6. Check the build logs to confirm Python 3.11 and successful dependency installation,
   then test the camera and sentence tools.

Community Cloud installs root `requirements.txt` and `packages.txt`. Do not install
`training/requirements.txt` into the deployed app.

**Existing deployment using Python 3.14.9:** a reboot or requirements change does
not change its Python interpreter. Streamlit's documented process is to record the
current subdomain, repository/branch/entry point, and secrets; delete the Cloud app;
then recreate it with the same settings and **Python 3.11** selected under
**Advanced settings**. Restore the secrets and reuse `asl-recognition-7` if available.
This causes downtime while the replacement deploys.

Python is selected in the Cloud interface. This repository does not rely on
`runtime.txt`, `.python-version`, or `.streamlit/config.toml` to force Cloud's
interpreter. See Streamlit's official [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [Python-version change procedure](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

### Configure camera connectivity

For restrictive networks, configure a Twilio TURN relay in **Settings → Secrets**
using [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example). For local
use, copy the template to `.streamlit/secrets.toml` and enter your credentials.
The real secrets file is ignored by Git.

Without TURN credentials the app falls back to public STUN, which may fail on some
mobile or corporate networks. Twilio relay usage can incur charges. If the public
page shows “Error running app,” open **Manage app** to inspect the detailed logs.

## Limitations and Future Improvements

- This is a pose classifier, not a continuous ASL translator. Motion-based letters
  such as J and Z may be less reliable without temporal modeling.
- Recognition depends on lighting, framing, camera quality, and available CPU.
  Per-session limits do not guarantee a particular number of concurrent users.
- Camera frames travel to the app server for processing. The app does not write
  video or transcripts to disk; session text may be lost on restart/reconnect.
- Selecting **Speak Sentence** sends the sentence to Google's speech service
  through gTTS and requires an internet connection.
- Potential improvements include temporal gesture recognition, evaluation across
  more users and lighting conditions, and measured capacity testing on the host.
