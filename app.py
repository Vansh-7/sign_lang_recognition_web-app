"""Real-time ASL recognition with bounded video work and per-session cleanup."""
import os

# Set these before importing numerical libraries to avoid CPU oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import logging
import threading
import time
from io import BytesIO
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from asl_model import ASLModel
from gtts import gTTS
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from twilio.rest import Client

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
MAX_SENTENCE_LENGTH = 2000
MAX_FRAME_WIDTH = 480
INFERENCE_INTERVAL = 0.1  # At most ten landmark/model evaluations per second.

st.set_page_config(page_title="ASL Recognition", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def get_ice_servers():
    # TURN credentials expire; don't cache a token for the life of the process.
    try:
        if "twilio" in st.secrets:
            config = st.secrets["twilio"]
            client = Client(config["account_sid"], config["auth_token"], timeout=10)
            return client.tokens.create(ttl=7200).ice_servers
    except Exception:
        LOGGER.warning("TURN credentials unavailable; using STUN.")
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


@st.cache_resource(show_spinner="Loading ASL model...")
def load_model():
    # Shared across reruns/sessions. No optimizer state or per-frame predict workers.
    cv2.setNumThreads(1)
    model = ASLModel(BASE_DIR / "asl_mediapipe_mlp_model.h5")
    model(np.zeros((1, 63), dtype=np.float32))
    return model, threading.Lock()


class ASLProcessor(VideoProcessorBase):
    def __init__(self, model, model_lock, sentence=""):
        self._model = model
        self._model_lock = model_lock
        self._lock = threading.Lock()
        self._hands = None  # Allocate only when the camera supplies its first frame.
        self._closed = False
        self._last_inference = float("-inf")
        self._landmarks = None
        self._error = None
        self.sentence = sentence[-MAX_SENTENCE_LENGTH:]
        self.last_prediction = None
        self.consecutive_frames = 0
        self.stability_threshold = 5
        self.cooldown_duration = 8
        self.cooldown = 0
        self.prediction_text = ""

    def configure(self, stability, cooldown):
        with self._lock:
            self.stability_threshold = stability
            self.cooldown_duration = cooldown

    def snapshot(self):
        with self._lock:
            return self.sentence, self._error

    def set_sentence(self, sentence):
        with self._lock:
            self.sentence = sentence[-MAX_SENTENCE_LENGTH:]
            self._reset_prediction()
            self.cooldown = 0

    def _reset_prediction(self):
        self.prediction_text = ""
        self.last_prediction = None
        self.consecutive_frames = 0

    def _accept_prediction(self, probabilities):
        if not np.all(np.isfinite(probabilities)) or np.max(probabilities) <= 0.8:
            self._reset_prediction()
            return
        sign = CLASSES[int(np.argmax(probabilities))]
        self.prediction_text = sign
        if self.cooldown > 0:
            return
        if sign == self.last_prediction:
            self.consecutive_frames += 1
        else:
            self.last_prediction = sign
            self.consecutive_frames = 1
        if self.consecutive_frames >= self.stability_threshold:
            if sign == "space":
                self.sentence += " "
            elif sign == "del":
                self.sentence = self.sentence[:-1]
            else:
                self.sentence += sign
            self.sentence = self.sentence[-MAX_SENTENCE_LENGTH:]
            self.consecutive_frames = 0
            self.cooldown = self.cooldown_duration

    def recv(self, frame):
        # Also protects cleanup and UI edits from racing with MediaPipe processing.
        with self._lock:
            if self._closed:
                return frame
            try:
                img = frame.to_ndarray(format="bgr24")
                height, width = img.shape[:2]
                if width > MAX_FRAME_WIDTH:
                    img = cv2.resize(
                        img, (MAX_FRAME_WIDTH, max(1, round(height * MAX_FRAME_WIDTH / width))),
                        interpolation=cv2.INTER_AREA,
                    )
                img = cv2.flip(img, 1)
                now = time.monotonic()
                if now - self._last_inference >= INFERENCE_INTERVAL:
                    self._last_inference = now
                    if self._hands is None:
                        self._hands = mp.solutions.hands.Hands(
                            static_image_mode=False, max_num_hands=1, model_complexity=0,
                            min_detection_confidence=0.7, min_tracking_confidence=0.7,
                        )
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = self._hands.process(rgb)
                    self._landmarks = (
                        results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
                    )
                    if self.cooldown > 0:
                        self.cooldown -= 1
                    if self._landmarks is None:
                        self._reset_prediction()
                    else:
                        features = np.asarray(
                            [[v for point in self._landmarks.landmark
                              for v in (point.x, point.y, point.z)]], dtype=np.float32,
                        )
                        with self._model_lock:
                            probabilities = self._model(features)[0]
                        self._accept_prediction(probabilities)
                    self._error = None

                if self._landmarks is not None:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img, self._landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
                cv2.putText(img, f"Sign: {self.prediction_text}", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                height, width = img.shape[:2]
                cv2.rectangle(img, (0, max(0, height - 40)), (width, height), (0, 0, 0), -1)
                cv2.putText(img, f"Sentence: {self.sentence[-24:]}", (12, height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                if self._error is None:
                    LOGGER.exception("ASL frame processing failed")
                self._error = "Recognition failed. Stop and restart the camera."
                self._reset_prediction()
                return frame

    async def recv_queued(self, frames):
        # Discard stale input instead of building latency under load.
        return [self.recv(frames[-1])]

    def on_ended(self):
        with self._lock:
            self._closed = True
            if self._hands is not None:
                self._hands.close()
                self._hands = None
            self._landmarks = None


def main():
    st.session_state.setdefault("asl_text", "")
    st.session_state.setdefault("last_processor_text", None)
    st.session_state.setdefault("speech_audio", None)
    st.title("🖐️ Real-Time ASL Recognition App")

    try:
        model, model_lock = load_model()
    except Exception:
        LOGGER.exception("Could not load ASL model")
        st.error("The ASL model could not be loaded. Check the model file and dependencies.")
        st.stop()

    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.subheader("Webcam Feed")
        st.markdown("Allow camera access, then hold a sign steady to add it to the sentence.")
        initial_sentence = st.session_state["asl_text"]
        webrtc_ctx = webrtc_streamer(
            key="asl-detection",
            video_processor_factory=lambda: ASLProcessor(model, model_lock, initial_sentence),
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
            media_stream_constraints={
                "video": {"width": {"ideal": 480, "max": 640},
                          "height": {"ideal": 360, "max": 480},
                          "frameRate": {"ideal": 15, "max": 15}},
                "audio": False,
            },
            async_processing=True,
        )
        with st.expander("⚙️ Sensitivity Settings"):
            st.caption("Recognition runs up to 10 times per second to keep CPU usage bounded.")
            stability = st.slider("Detection Speed (recognition samples to hold)", 3, 30, 5)
            cooldown = st.slider("Cooldown (recognition samples to wait)", 5, 30, 8)
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.configure(stability, cooldown)

    with col2:
        st.subheader("ASL Reference")
        if (BASE_DIR / "ASL.png").exists():
            st.image(str(BASE_DIR / "ASL.png"), caption="ASL Alphabet", use_container_width=True)
        else:
            st.warning("ASL.png not found.")


    def edit_sentence():
        st.session_state["asl_text"] = st.session_state["sentence_editor"]
        st.session_state["speech_audio"] = None
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_sentence(st.session_state["asl_text"])
            st.session_state["last_processor_text"] = st.session_state["asl_text"]


    def clear_sentence():
        st.session_state["asl_text"] = ""
        st.session_state["sentence_editor"] = ""
        st.session_state["last_processor_text"] = ""
        st.session_state["speech_audio"] = None
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_sentence("")


    # Only this small panel refreshes; no blocking loop or full-app reruns per letter.
    @st.fragment(run_every=0.5)
    def sentence_tools():
        processor = webrtc_ctx.video_processor
        if processor:
            live_sentence, error = processor.snapshot()
            if live_sentence != st.session_state["last_processor_text"]:
                st.session_state["asl_text"] = live_sentence
                st.session_state["last_processor_text"] = live_sentence
                st.session_state["speech_audio"] = None
            if error:
                st.warning(error)
        st.subheader("Sentence Tools")
        st.session_state["sentence_editor"] = st.session_state["asl_text"]
        sentence = st.text_area("Prediction:", key="sentence_editor", height=100,
                                max_chars=MAX_SENTENCE_LENGTH, on_change=edit_sentence)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔊 Speak Sentence", disabled=not sentence.strip()):
                try:
                    sound_file = BytesIO()
                    gTTS(text=sentence, lang="en", timeout=10).write_to_fp(sound_file)
                    st.session_state["speech_audio"] = sound_file.getvalue()
                except Exception:
                    LOGGER.exception("Audio generation failed")
                    st.error("Audio generation failed. Please try again.")
        with c2:
            st.download_button("💾 Save as .txt", data=sentence,
                               file_name="asl_translation.txt", mime="text/plain",
                               disabled=not sentence)
        with c3:
            st.button("🧹 Clear Text", on_click=clear_sentence)
        if st.session_state["speech_audio"]:
            st.audio(st.session_state["speech_audio"], format="audio/mp3")


    st.divider()
    sentence_tools()


if __name__ == "__main__":
    main()
