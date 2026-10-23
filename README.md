# 🖐️ Real-Time ASL Recognition

> A powerful real-time American Sign Language (ASL) detection system built with **Streamlit**, **MediaPipe**, and **TensorFlow**. This application translates hand gestures into text and speech instantly using a lightweight neural network.

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
  * **Machine Learning:** TensorFlow (CPU), Keras
  * **Audio Processing:** gTTS (Google Text-to-Speech)
  * **Data Processing:** NumPy, Pandas

## 📂 Project Structure

```bash
├── app.py                            # Main Streamlit application script
├── asl_mediapipe_mlp_model.h5        # Pre-trained TensorFlow/Keras model
├── requirements.txt                  # Python dependencies
├── packages.txt                      # OS-level dependencies (for deployment)
├── ASL.png                           # Reference image for ASL alphabet
├── asl_mediapipe_final.ipynb         # Notebook used for training the model
└── asl_mediapipe_keypoints_dataset.csv # Dataset of hand landmarks
```

## ⚙️ Installation

### Prerequisites

Use **Python 3.11** (including in Streamlit Community Cloud advanced settings).

### 1\. Clone the Repository

```bash
git clone https://github.com/Vansh-7/sign_lang_recognition_web-app.git
cd sign_lang_recognition_web-app
```

### 2\. Create a Virtual Environment

It is recommended to use a virtual environment to avoid conflicts.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3\. Install Dependencies

Install the required libraries from:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Once the app launches in your browser:

1.  **Grant Camera Access:** Click "Start" on the webcam feed.
2.  **Perform Signs:** Hold your hand up to the camera. The model detects landmarks and predicts the letter.
3.  **Adjust Settings:** Use the "Sensitivity Settings" dropdown to tune:
      * *Detection Speed:* How many frames to hold a sign before it registers.
      * *Cooldown:* How long to wait between letters.
4.  **Tools:** Use the buttons to Speak the sentence, Save it to a file, or Clear the text.

## 🧠 How It Works

1.  **Video Capture:** The app uses `streamlit-webrtc` to capture video frames directly in the browser.
2.  **Landmark Extraction:** **MediaPipe Hands** processes each frame to extract 21 hand landmarks (consisting of x, y, and z coordinates).
3.  **Preprocessing:** These coordinates are flattened into a vector and fed into the neural network.
4.  **Inference:** The loaded `asl_mediapipe_mlp_model.h5` predicts the character (A-Z, Space, Del) with a confidence threshold of 0.8.
5.  **Post-Processing:** Logic handles repetitive frames to ensure stable sentence construction (e.g., requiring 5 consecutive frames of the same sign).

## 📜 License

This project is open-source and available under the **MIT License**.

## Streamlit resource usage

The web app reads the existing H5 model with `h5py` and runs its three dense
layers with NumPy. TensorFlow and Keras are only needed for training in the
notebook; they are no longer web app dependencies. Keep `asl_model.py` beside
`app.py` and the existing model file. No retraining or model conversion is needed.

Camera capture is limited to 15 FPS with a preferred 480 × 360 resolution.
Recognition runs at most 10 times per second using MediaPipe's lightweight hand
tracker, and stale queued frames are dropped. Sensitivity sliders count
recognition samples (five samples take at least about 0.5 seconds). Sentence
updates refresh only the tools panel every 0.5 seconds. Each camera session
closes its own tracker on disconnect. Sentence history is capped at 2,000
characters, with the last 24 visible in the video overlay.

After merging these changes, reboot the Streamlit app so its environment is
rebuilt using the updated requirements. Configure Twilio TURN credentials in
Streamlit secrets for networks that cannot connect through the STUN fallback.
TURN credentials refresh hourly.

Development checks: `python -m pip install pytest`, then `python -m pytest -q`.
A live browser camera test on the deployed host is still needed to assess
network latency and concurrency on that host.
