"""Shared model resources and expiring WebRTC connectivity configuration."""

import logging
import threading

import cv2
import numpy as np
import streamlit as st
from twilio.rest import Client

from asl_recognition.config import MODEL_PATH
from asl_recognition.model import ASLModel

LOGGER = logging.getLogger(__name__)


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
    model = ASLModel(MODEL_PATH)
    model(np.zeros((1, 63), dtype=np.float32))
    return model, threading.Lock()
