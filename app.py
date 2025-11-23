import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import av
import time
from gtts import gTTS
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. Setup & Configuration ---
st.set_page_config(page_title="ASL Recognition Pro", layout="wide")

# Google STUN server to fix connection issues on Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load Model (Cached to prevent reloading on every frame)
@st.cache_resource
def load_model():
    try:
        # Standard Keras 3 loading (matches the 'modern stack' requirements)
        return tf.keras.models.load_model('asl_mediapipe_mlp_model.h5')
    except Exception as e:
        st.error(f"Error loading model. Please ensure 'asl_mediapipe_mlp_model.h5' is in the repo. Error: {e}")
        return None

model = load_model()

# Define Classes (Must match the encoder in your Jupyter notebook)
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'del', 'space']

# MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- 2. Video Processing Logic ---
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        # Variables for sentence formation logic
        self.sentence = ""
        self.last_prediction = None
        self.consecutive_frames = 0
        self.stability_threshold = 15  # Frames to hold a sign before registering it
        self.cooldown = 0
        self.prediction_text = ""

    def recv(self, frame):
        try:
            # Convert frame to format OpenCV understands
            img = frame.to_ndarray(format="bgr24")
            
            # Flip horizontally for a mirror effect & Convert to RGB
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = hands.process(img_rgb)
            
            current_sign = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract x, y, z coordinates for model input
                    data_aux = []
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x)
                        data_aux.append(hand_landmarks.landmark[i].y)
                        data_aux.append(hand_landmarks.landmark[i].z)
                    
                    # Make Prediction
                    if model is not None:
                        try:
                            input_data = np.array([data_aux])
                            prediction = model.predict(input_data, verbose=0)
                            predicted_index = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            if confidence > 0.8: # Confidence Threshold
                                current_sign = CLASSES[predicted_index]
                                self.prediction_text = current_sign
                                
                                # --- Sentence Construction Logic ---
                                if self.cooldown > 0:
                                    self.cooldown -= 1
                                else:
                                    if current_sign == self.last_prediction:
                                        self.consecutive_frames += 1
                                        # If sign is held stable for enough frames...
                                        if self.consecutive_frames >= self.stability_threshold:
                                            if current_sign == 'space':
                                                self.sentence += " "
                                            elif current_sign == 'del':
                                                self.sentence = self.sentence[:-1]
                                            else:
                                                self.sentence += current_sign
                                            
                                            # Reset and start cooldown
                                            self.consecutive_frames = 0
                                            self.cooldown = 20 # Wait frames before accepting next input
                                    else:
                                        self.last_prediction = current_sign
                                        self.consecutive_frames = 0
                        except Exception as e:
                            # Fail gracefully if prediction crashes
                            print(f"Prediction Error: {e}")

            # --- UI Overlays (Burned into the video feed) ---
            
            # 1. Display Current Sign (Top Left)
            cv2.putText(img, f"Sign: {self.prediction_text}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # 2. Display Formed Sentence (Bottom Banner)
            h, w, _ = img.shape
            # Black rectangle background
            cv2.rectangle(img, (0, h-60), (w, h), (0, 0, 0), -1)
            # Sentence text
            cv2.putText(img, f"Sentence: {self.sentence}", (20, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # If critical error, print it but return original frame so stream doesn't die
            print(f"Frame processing error: {e}")
            return frame

# --- 3. Streamlit App Layout ---
st.title("🖐️ Real-Time ASL Recognition App")

col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.subheader("Webcam Feed")
    st.markdown("""
    **Instructions:**
    1. Allow camera access.
    2. Hold a sign stable for **1 second** to add it.
    3. Use **'space'** gesture for space, **'del'** to delete.
    """)
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="asl-detection",
        video_processor_factory=ASLProcessor,
        mode="sendrecv",
        rtc_configuration=RTC_CONFIGURATION, # Fixes cloud connection issues
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("ASL Reference")
    try:
        st.image("ASL.png", caption="ASL Alphabet", use_container_width=True)
    except:
        st.warning("⚠️ 'ASL.png' not found in repository.")

# --- 4. Interactive Tools (Speech & Save) ---
st.divider()
st.subheader("📝 Sentence Tools")

# Text Area to edit/view the sentence
# Note: We can't pull live data OUT of the video processor easily in Streamlit.
# So we ask the user to type what they see if they want to save/speak it.
sentence_input = st.text_area("Type the sentence you formed above to Speak or Save:", height=70)

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

# Feature 3: Info
with c3:
    st.info("Tip: The sentence is built automatically in the video black bar!")