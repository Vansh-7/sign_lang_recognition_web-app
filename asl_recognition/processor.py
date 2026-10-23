"""Per-camera landmark tracking, bounded inference, and synchronized sentence state."""

import logging
import threading
import time

import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from asl_recognition.config import (
    CLASSES,
    CONFIDENCE_THRESHOLD,
    INFERENCE_INTERVAL,
    MAX_FRAME_WIDTH,
    MAX_SENTENCE_LENGTH,
)

LOGGER = logging.getLogger(__name__)


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
        if not np.all(np.isfinite(probabilities)) or np.max(probabilities) <= CONFIDENCE_THRESHOLD:
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
                        img,
                        (MAX_FRAME_WIDTH, max(1, round(height * MAX_FRAME_WIDTH / width))),
                        interpolation=cv2.INTER_AREA,
                    )
                img = cv2.flip(img, 1)
                now = time.monotonic()
                if now - self._last_inference >= INFERENCE_INTERVAL:
                    self._last_inference = now
                    if self._hands is None:
                        self._hands = mp.solutions.hands.Hands(
                            static_image_mode=False,
                            max_num_hands=1,
                            model_complexity=0,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7,
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
                            [
                                [
                                    v
                                    for point in self._landmarks.landmark
                                    for v in (point.x, point.y, point.z)
                                ]
                            ],
                            dtype=np.float32,
                        )
                        with self._model_lock:
                            probabilities = self._model(features)[0]
                        self._accept_prediction(probabilities)
                    self._error = None

                if self._landmarks is not None:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img, self._landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
                cv2.putText(
                    img,
                    f"Sign: {self.prediction_text}",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                height, width = img.shape[:2]
                cv2.rectangle(img, (0, max(0, height - 40)), (width, height), (0, 0, 0), -1)
                cv2.putText(
                    img,
                    f"Sentence: {self.sentence[-24:]}",
                    (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
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
