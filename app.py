import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import av
import time
from gtts import gTTS
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- 1. Setup & Config ---
st.set_page_config(page_title="ASL Recognition Pro", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('asl_mediapipe_mlp_model.h5')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Classes (Ensure these match your training order exactly)
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'del', 'space']

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- 2. Video Processor Class (The Brains) ---
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.sentence = ""
        self.last_prediction = None
        self.consecutive_frames = 0
        self.stability_threshold = 15  # Number of frames to hold a sign before adding
        self.cooldown = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip and Convert
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        
        prediction_text = ""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract Landmarks (x, y, z)
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x)
                    data_aux.append(hand_landmarks.landmark[i].y)
                    data_aux.append(hand_landmarks.landmark[i].z)
                
                # Prediction
                try:
                    input_data = np.array([data_aux])
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    if confidence > 0.8:
                        prediction_text = CLASSES[predicted_index]
                        
                        # --- Sentence Construction Logic ---
                        if self.cooldown > 0:
                            self.cooldown -= 1
                        else:
                            if prediction_text == self.last_prediction:
                                self.consecutive_frames += 1
                                if self.consecutive_frames >= self.stability_threshold:
                                    # Valid Gesture Detected
                                    if prediction_text == 'space':
                                        self.sentence += " "
                                    elif prediction_text == 'del':
                                        self.sentence = self.sentence[:-1]
                                    else:
                                        self.sentence += prediction_text
                                    
                                    # Reset
                                    self.consecutive_frames = 0
                                    self.cooldown = 20 # Wait 20 frames before next input
                            else:
                                self.last_prediction = prediction_text
                                self.consecutive_frames = 0
                                
                except Exception as e:
                    print(f"Error: {e}")

        # --- UI Overlays on Video ---
        # 1. Current Prediction (Top Left)
        cv2.putText(img, f"Sign: {prediction_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # 2. Formed Sentence (Bottom Area)
        # Create a black banner at the bottom
        h, w, _ = img.shape
        cv2.rectangle(img, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"Sentence: {self.sentence}", (20, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. Streamlit UI Layout ---
st.title("🖐️ Real-Time ASL Recognition App")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam Feed")
    st.info("Hold a sign for 1-2 seconds to add it to the sentence.")
    
    webrtc_ctx = webrtc_streamer(
    key="asl-detection",
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

with col2:
    st.subheader("ASL Reference")
    try:
        st.image("ASL.png", caption="ASL Alphabet", use_column_width=True)
    except:
        st.warning("ASL.png not found. Please upload it.")

# --- 4. Tools: Speech & Save ---
st.divider()
st.subheader("📝 Tools")

# Input for correction (User copies what they see on video)
sentence_input = st.text_area("Type the sentence formed above to Save or Speak:", height=70)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("🔊 Speak Sentence"):
        if sentence_input:
            try:
                tts = gTTS(text=sentence_input, lang='en')
                sound_file = BytesIO()
                tts.write_to_fp(sound_file)
                st.audio(sound_file)
            except Exception as e:
                st.error(f"Audio error: {e}")
        else:
            st.warning("Enter text to speak.")

with c2:
    if sentence_input:
        st.download_button(
            label="💾 Save to File",
            data=sentence_input,
            file_name="asl_translation.txt",
            mime="text/plain"
        )
    else:
        st.button("💾 Save to File", disabled=True)

with c3:
    if st.button("🧹 Clear Text"):
        # We can't clear the video processor state easily from here without complex session state sync
        # But we can clear the text box
        pass # Streamlit rerun clears the UI logic naturally