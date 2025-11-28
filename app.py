import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import av
import time
from gtts import gTTS
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import streamlit_webrtc.shutdown
import threading
import asyncio
from aioice import stun
import logging

# --- BEGIN PATCH ---
# Fix for "NoneType object has no attribute sendto" on reload
# We must use the mangled name '_Transaction__retry' because it is a private method.
try:
    if hasattr(stun.Transaction, "_Transaction__retry"):
        original_retry = stun.Transaction._Transaction__retry

        def patched_retry(self):
            try:
                original_retry(self)
            except (AttributeError, OSError) as e:
                # Catch the specific race condition error when connection dies
                if "NoneType" in str(e) or "sendto" in str(e):
                    pass 
                else:
                    raise e

        stun.Transaction._Transaction__retry = patched_retry
except Exception as e:
    logging.warning(f"Could not patch aioice.stun.Transaction: {e}")
# --- END PATCH ---


try:
    loop = asyncio.get_event_loop()
    
    def custom_exception_handler(loop, context):
        exception = context.get('exception')
        # Swallow the specific "NoneType" errors that happen on disconnect
        if isinstance(exception, AttributeError) and (
            "'NoneType' object has no attribute 'sendto'" in str(exception) or 
            "'NoneType' object has no attribute 'call_exception_handler'" in str(exception)
        ):
            return 
        # Otherwise, handle normally
        loop.default_exception_handler(context)

    loop.set_exception_handler(custom_exception_handler)
except Exception:
    pass

_original_stop = streamlit_webrtc.shutdown.SessionShutdownObserver.stop

def _safe_stop(self):
    if getattr(self, "_polling_thread", None) is None:
        return
    _original_stop(self)

streamlit_webrtc.shutdown.SessionShutdownObserver.stop = _safe_stop
# ---------------------------------------------------------

# 1. Setup & Configuration
st.set_page_config(page_title="ASL Recognition", layout="wide")

if "asl_text" not in st.session_state:
    st.session_state["asl_text"] = ""

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]}
)

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('asl_mediapipe_mlp_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'del', 'space']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2. Video Processing Logic
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.sentence = ""
        self.last_prediction = None
        self.consecutive_frames = 0
        # Default values (will be updated by sliders)
        self.stability_threshold = 5 
        self.cooldown_duration = 8
        self.cooldown = 0
        self.prediction_text = ""

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    data_aux = []
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x)
                        data_aux.append(hand_landmarks.landmark[i].y)
                        data_aux.append(hand_landmarks.landmark[i].z)
                    
                    if model is not None:
                        try:
                            input_data = np.array([data_aux])
                            prediction = model.predict(input_data, verbose=0)
                            predicted_index = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            if confidence > 0.8:
                                current_sign = CLASSES[predicted_index]
                                self.prediction_text = current_sign
                                
                                if self.cooldown > 0:
                                    self.cooldown -= 1
                                else:
                                    if current_sign == self.last_prediction:
                                        self.consecutive_frames += 1
                                        # Use dynamic threshold from sliders
                                        if self.consecutive_frames >= self.stability_threshold:
                                            if current_sign == 'space':
                                                self.sentence += " "
                                            elif current_sign == 'del':
                                                self.sentence = self.sentence[:-1]
                                            else:
                                                self.sentence += current_sign
                                            
                                            self.consecutive_frames = 0
                                            self.cooldown = self.cooldown_duration
                                    else:
                                        self.last_prediction = current_sign
                                        self.consecutive_frames = 0
                        except:
                            pass

            # Overlays
            cv2.putText(img, f"Sign: {self.prediction_text}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            h, w, _ = img.shape
            cv2.rectangle(img, (0, h-60), (w, h), (0, 0, 0), -1)
            cv2.putText(img, f"Sentence: {self.sentence}", (20, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception:
            return frame

# 3. Streamlit UI
st.title("🖐️ Real-Time ASL Recognition App")

col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.subheader("Webcam Feed")
    st.markdown("""
    **Instructions:**
    1. Allow camera access.
    2. Hold a sign stable to add it (Adjust speed in settings below).
    """)

    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="asl-detection",
        video_processor_factory=ASLProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Performance Tuners
    with st.expander("⚙️ Sensitivity Settings"):
        stability = st.slider("Detection Speed (Frames to hold)", 3, 30, 5, help="Lower = Faster detection, Higher = More accurate")
        cooldown = st.slider("Cooldown (Frames wait)", 5, 30, 8, help="Wait time between letters")

    # Update processor settings dynamically
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.stability_threshold = stability
        webrtc_ctx.video_processor.cooldown_duration = cooldown

with col2:
    st.subheader("ASL Reference")
    try:
        st.image("ASL.png", caption="ASL Alphabet", use_container_width=True)
    except:
        st.warning("⚠️ 'ASL.png' not found.")

st.divider()
st.subheader("Sentence Tools")

# Text Area & Tools
text_placeholder = st.empty()
sentence_input = text_placeholder.text_area("Prediction:", value=st.session_state["asl_text"], height=70)

c1, c2, c3 = st.columns(3)

# Feature 1: Text to Speech
with c1:
    if st.button("🔊 Speak Sentence"):
        if sentence_input:
            try:
                tts = gTTS(text=sentence_input, lang='en')
                sound_file = BytesIO()
                tts.write_to_fp(sound_file)
                st.audio(sound_file)
            except Exception as e:
                st.error(f"Audio generation error: {e}")
        else:
            st.warning("Please enter text to speak.")

# Feature 2: Save to File
with c2:
    if sentence_input:
        st.download_button(
            label="💾 Save as .txt",
            data=sentence_input,
            file_name="asl_translation.txt",
            mime="text/plain"
        )
    else:
        st.button("💾 Save as .txt", disabled=True)

# Feature 3: Clear Text
with c3:
    if st.button("🧹 Clear Text"):
        # Clear session state
        st.session_state["asl_text"] = ""
        # Clear the video processor memory if it's running
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.sentence = ""
        st.rerun()

# 4. Background Sync Loop
if webrtc_ctx.state.playing:
    # Create a placeholder container once, so we don't need to full rerun just to update text
    # Note: Streamlit execution model usually requires rerun to update UI outside the loop, 
    # but we can check strictly for changes.
    
    while webrtc_ctx.state.playing:
        if webrtc_ctx.video_processor:
            live_sentence = webrtc_ctx.video_processor.sentence
            
            # ONLY rerun if the text has actually changed
            if live_sentence != st.session_state["asl_text"]:
                st.session_state["asl_text"] = live_sentence
                st.rerun()
                
        # Increase sleep slightly to relieve CPU and reduce race conditions
        time.sleep(0.2)