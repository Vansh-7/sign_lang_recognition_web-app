# 🖐️ Real-Time ASL Recognition

> A real-time American Sign Language (ASL) recognition system built with **Streamlit**, **MediaPipe**, and a lightweight neural network. Turn hand gestures into editable text and speech.

**🌐 [Try the live app](https://asl-recognition-7.streamlit.app/)**

## Features

  * **Real-Time Detection:** Uses **Streamlit WebRTC** to process live webcam feeds with low latency.
  * **Smart Prediction:** Filters noise using a customizable stability algorithm (sliders for detection speed and cooldown).
  * **Text-to-Speech (TTS):** Converts the predicted sign language sentence into spoken audio using **gTTS**.
  * **Save & Export:** Allows users to download the translated sentence as a `.txt` file.
  * **Interactive UI:** Includes an integrated ASL alphabet reference chart and dynamic sensitivity settings.
  * **Lightweight Model:** Uses a customized Multi-Layer Perceptron (MLP) trained on MediaPipe hand landmarks (x, y, z coordinates).

## Tech Stack

  * **Frontend:** Streamlit
  * **Computer Vision:** MediaPipe, OpenCV, Streamlit-WebRTC
  * **Model Inference:** NumPy and h5py (TensorFlow/Keras are used only for training)
  * **Audio Processing:** gTTS (Google Text-to-Speech)
  * **Data Processing:** NumPy

## 📂 Project Structure

```text
├── app.py                            # Streamlit UI and webcam processing
├── asl_model.py                      # Lightweight model loader and inference
├── asl_mediapipe_mlp_model.h5        # Pre-trained model
├── ASL.png                           # ASL alphabet reference
├── requirements.txt                  # App dependencies
├── packages.txt                      # Linux dependencies for deployment
├── .streamlit/                       # Configuration and secrets example
├── training/                         # Original notebook, landmark CSV, and training dependencies
└── tests/                            # App and deployment checks
```

## ⚙️ Installation

### Prerequisites

Use **Python 3.11** and a browser with webcam access.

### 1\. Clone the Repository

```bash
git clone https://github.com/Vansh-7/sign_lang_recognition_web-app.git
cd sign_lang_recognition_web-app
```

### 2\. Create a Virtual Environment

It is recommended to use a virtual environment to avoid conflicts.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 3\. Install Dependencies

Install the app dependencies:

```bash
python -m pip install -r requirements.txt
```

## ▶️ Usage

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Once the app launches in your browser:

1.  **Grant Camera Access:** Click "Start" on the webcam feed.
2.  **Perform Signs:** Hold your hand up to the camera. The model detects landmarks and predicts the letter.
3.  **Adjust Settings:** Use the "Sensitivity Settings" dropdown to tune:
      * *Detection Speed:* How many recognition samples to hold a sign before it registers.
      * *Cooldown:* How long to wait between letters.
4.  **Tools:** Edit the text, Speak the sentence, Save it to a file, or Clear it.
5.  **Stop:** Stop the camera when finished to release its resources.

## ☁️ Streamlit Deployment

Deploy from this repository with **`app.py`** as the main file and **Python 3.11**.
Keep the model, `ASL.png`, and app files together. Community Cloud installs
`requirements.txt` and `packages.txt`; allow it to rebuild after dependency changes.

For reliable camera connections across networks, add Twilio credentials in
**Settings → Secrets**, following [the secrets example](.streamlit/secrets.toml.example).
Locally, copy the example to `.streamlit/secrets.toml` and fill in your credentials.
The real secrets file is ignored by Git. Without TURN credentials, the STUN
fallback may fail on restrictive networks. Twilio relay usage can incur charges.

If the hosted app shows an error, check **Manage app** for the detailed logs and
reboot after fixing the reported issue.

## 🧠 How It Works

1.  **Video Capture:** The app uses `streamlit-webrtc` to capture video frames directly in the browser.
2.  **Landmark Extraction:** **MediaPipe Hands** processes each frame to extract 21 hand landmarks (consisting of x, y, and z coordinates).
3.  **Preprocessing:** These coordinates are flattened into a vector and fed into the neural network.
4.  **Inference:** The existing `.h5` model runs with NumPy and h5py, predicting A–Z, Space, or Del above 0.8 confidence. The deployed app does not need TensorFlow.
5.  **Post-Processing:** Stability and cooldown settings filter repeated predictions before adding them to the sentence.

Capture is capped at 15 FPS and recognition at 10 samples per second. The app
drops stale frames, refreshes the sentence panel independently, and closes the
hand tracker when the camera stops.

The model recognizes individual hand poses rather than continuous ASL sentences;
motion-based letters such as J and Z may be less reliable. Camera frames are
processed on the server; selecting Speak sends the sentence to Google through gTTS.

### Training and Testing

The original notebook and landmark dataset are in `training/`. Use
`training/requirements.txt` in a separate environment for model work. The notebook
can use the supplied CSV; extracting landmarks from new images requires your
own dataset, configured through `ASL_TRAINING_IMAGES`.

To run the app checks: `python -m pip install pytest`, then `python -m pytest -q`.
Also verify the camera and network connection on the deployed app.

## 📜 License

This project is open-source and available under the **MIT License**.
