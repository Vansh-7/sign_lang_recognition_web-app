import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class FixedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        # Keras 3 uses 'batch_shape', but Keras 2 expects 'batch_input_shape'
        if batch_shape is not None:
            kwargs['batch_input_shape'] = batch_shape
        super().__init__(**kwargs)

# 1. Load Model & Setup MediaPipe
@st.cache_resource
def load_model():
    # We pass the custom layer to handle the version mismatch
    return tf.keras.models.load_model(
        'asl_mediapipe_mlp_model.h5', 
        custom_objects={'InputLayer': FixedInputLayer}
    )

model = load_model()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2. Define the Classes (Must match your training encoder)
# REPLACE THIS LIST with the actual classes from your 'encoder.classes_'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'del', 'space'] 

# 3. Define Video Processor
class ASLProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip and convert to RGB for MediaPipe
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        
        prediction_text = "Waiting for hand..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # --- YOUR PREPROCESSING LOGIC HERE ---
                # Extract x, y, z coordinates into a flat list
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x)
                    data_aux.append(hand_landmarks.landmark[i].y)
                    data_aux.append(hand_landmarks.landmark[i].z) # Remove this line if your model trained on x,y only
                
                # Prediction
                try:
                    # Reshape input to match model (1, 63) or (1, 42)
                    input_data = np.array([data_aux])
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    prediction_text = CLASSES[predicted_index]
                    confidence = np.max(prediction)
                    
                    # Display text on screen
                    if confidence > 0.8: # Threshold
                         cv2.putText(img, f"Sign: {prediction_text}", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                except Exception as e:
                    print(f"Prediction Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Streamlit UI
st.title("🖐️ Real-Time ASL Recognition")
st.text("Uses MediaPipe + Keras MLP")

webrtc_streamer(
    key="asl-detection",
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)