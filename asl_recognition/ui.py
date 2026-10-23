"""Streamlit page and fragment-based sentence tools."""

import logging
from io import BytesIO

import streamlit as st
from gtts import gTTS
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from asl_recognition.config import MAX_SENTENCE_LENGTH, REFERENCE_IMAGE_PATH
from asl_recognition.processor import ASLProcessor
from asl_recognition.services import get_ice_servers, load_model

LOGGER = logging.getLogger(__name__)


def main():
    st.set_page_config(page_title="ASL Recognition", layout="wide")
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
                "video": {
                    "width": {"ideal": 480, "max": 640},
                    "height": {"ideal": 360, "max": 480},
                    "frameRate": {"ideal": 15, "max": 15},
                },
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
        if REFERENCE_IMAGE_PATH.exists():
            st.image(str(REFERENCE_IMAGE_PATH), caption="ASL Alphabet", use_container_width=True)
        else:
            st.warning("The ASL reference image is unavailable.")

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
        sentence = st.text_area(
            "Prediction:",
            key="sentence_editor",
            height=100,
            max_chars=MAX_SENTENCE_LENGTH,
            on_change=edit_sentence,
        )
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
            st.download_button(
                "💾 Save as .txt",
                data=sentence,
                file_name="asl_translation.txt",
                mime="text/plain",
                disabled=not sentence,
            )
        with c3:
            st.button("🧹 Clear Text", on_click=clear_sentence)
        if st.session_state["speech_audio"]:
            st.audio(st.session_state["speech_audio"], format="audio/mp3")

    st.divider()
    sentence_tools()
