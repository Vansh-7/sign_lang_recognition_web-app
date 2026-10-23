# ASL Recognition

Recognize ASL alphabet hand poses from a browser camera, build editable text,
and export it as a text file or spoken audio.

**[Open the deployed app](https://asl-recognition-7.streamlit.app/)** ·
[Deployment guide](docs/deployment.md) · [Architecture](docs/architecture.md)

The app uses Streamlit, WebRTC, and MediaPipe for video and hand tracking.
A small neural network runs with NumPy using the existing H5 weights;
TensorFlow is only needed in the separate training environment.

## Use the app

1. Open the app, select **Start**, and allow camera access.
2. Keep one hand visible in good lighting and hold a sign steady.
3. Adjust detection speed and cooldown under **Sensitivity Settings**.
4. Edit the recognized text, speak it, download it, or clear it.
5. Select **Stop** when finished to release the camera and tracker.

The model predicts 28 classes: A–Z, `del`, and `space`. It classifies individual
hand poses; it does not translate continuous ASL sentences or interpret motion
sequences. Recognition of motion-based letters such as J and Z is limited.

## Run locally

Use **Python 3.11** and run these commands from the repository root:

```bash
git clone https://github.com/Vansh-7/sign_lang_recognition_web-app.git
cd sign_lang_recognition_web-app
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Debian/Ubuntu, install the native libraries listed in `packages.txt` if
OpenCV reports missing shared libraries. On other platforms, use their native
equivalents. For local automatic reloads, add `--server.fileWatcherType=auto`
to the run command; file watching is disabled by default for deployment.

## Deploy on Streamlit Community Cloud

| Setting | Value |
| --- | --- |
| Repository | `Vansh-7/sign_lang_recognition_web-app` |
| Branch | `main` after merging the deployment changes |
| Main file path | `app.py` |
| Python | `3.11` |
| App URL | `https://asl-recognition-7.streamlit.app/` |

The root `requirements.txt`, `packages.txt`, and `.streamlit/config.toml`
configure deployment. Keep `app.py` at the repository root. When testing the
unmerged changes, select `fix/streamlit-resource-usage` as the branch.

For reliable remote camera connections, configure TURN credentials in the
app's **Settings → Secrets** using `.streamlit/secrets.toml.example`. Locally,
copy that example to `.streamlit/secrets.toml` and fill in the values. Never
commit the real credentials. Without them, the app falls back to a public
STUN server, which may fail on restrictive networks. Twilio relay usage can
incur charges.

See the [deployment guide](docs/deployment.md) for updates, troubleshooting,
rollback, and post-deployment verification.

## Repository layout

```text
.
├── app.py                      # Stable Streamlit entry point
├── asl_recognition/
│   ├── config.py               # Resource limits and absolute asset paths
│   ├── model.py                # Read-only H5 weights and NumPy inference
│   ├── processor.py            # Per-camera tracking, prediction, cleanup
│   ├── services.py             # Cached model and expiring TURN credentials
│   └── ui.py                   # Streamlit page and sentence tools
├── assets/asl_alphabet.png
├── models/asl_mediapipe_mlp_model.h5
├── data/                       # Landmark dataset; unused at runtime
├── notebooks/                  # Training notebook without saved outputs
├── training/                   # Separate training dependencies and notes
├── tests/                      # Model, video, UI, and deployment checks
├── docs/                       # Deployment and architecture documentation
├── .streamlit/                 # App configuration and secrets example
├── .github/workflows/ci.yml    # Linux lint and regression checks
├── requirements.txt            # Pinned runtime dependencies
├── requirements-dev.txt        # Runtime plus test/lint tools
├── packages.txt                # Native Linux runtime dependencies
└── pyproject.toml              # Test and lint configuration
```

## Performance and data handling

- Capture: up to 15 FPS, preferably 480 × 360; processed frames are at most
  480 pixels wide. Recognition runs at most 10 times per second.
- When input falls behind, queued stale frames are dropped. The sentence
  panel refreshes every 0.5 seconds without rerunning the whole page.
- Each camera session owns a tracker and releases it on disconnect. Model
  weights are shared and loaded once per server process.
- Text is limited to 2,000 characters; the video overlay shows the last 24.
  Sliders count recognition samples, so five samples take at least about 0.5 s.
- Camera frames are sent to the app server for processing. The application
  does not deliberately save video frames or transcripts to disk. Session
  text and generated audio are temporary and may be lost on reconnect/reboot.
- Selecting **Speak Sentence** sends the sentence to Google's speech service
  through gTTS. TURN, when configured, relays the camera connection via Twilio.

These bounds reduce work per session; they do not guarantee unlimited
concurrent users. Verify latency and memory with the expected workload on
the deployed host.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI runs these checks on Python 3.11/Linux without TensorFlow installed.
Tests cover inference, frame limits, queue behavior, cleanup, session
isolation, text synchronization, and deployment file paths. Camera permissions,
real hand recognition, TURN connectivity, and multi-user capacity need a
browser check on the deployed app.

For model work, use the [separate training environment](training/README.md).
The dataset and notebook are preserved under `data/` and `notebooks/` and
are not loaded during a camera session.

## License

A license file has not been added to this repository. Confirm the permissions
for the code, model, dataset, and reference image before redistributing them.
