import streamlit as st
import os
from model import predict_emotion
import tempfile
import shutil

st.title("🎙️ Voice Emotion Detection")
st.write("Upload a `.wav` file to detect the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

if uploaded_file is not None:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(tmp_path, format="audio/wav")

    # Predict emotion
    with st.spinner("Analyzing..."):
        emotion = predict_emotion(tmp_path)

    st.success(f"Predicted Emotion: **{emotion}**")

    # Clean up temp file
    os.remove(tmp_path)
