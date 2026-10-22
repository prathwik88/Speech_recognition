
import io
import numpy as np
import librosa
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Speech Emotion AI",
    page_icon="🎙️",
    layout="centered"
)

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("best_model.keras")
    data = np.load("preprocessing.npz")

    mean = data["mean"]
    std = np.maximum(data["std"], 1e-6)
    classes = data["classes"]

    return model, mean, std, classes

model, mean, std, classes = load_resources()

def predict_emotion(audio_bytes):
    signal, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True
    )

    if len(signal) < 1600:
        raise ValueError("Please record a longer audio sample.")

    features = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40,
        n_fft=512,
        hop_length=256
    ).T

    max_len = 174

    if len(features) < max_len:
        features = np.pad(
            features,
            ((0, max_len - len(features)), (0, 0))
        )
    else:
        features = features[:max_len]

    features = (features - mean) / std
    features = features[np.newaxis, ...]

    probabilities = model.predict(features, verbose=0)[0]
    index = int(np.argmax(probabilities))

    return classes[index], probabilities

st.title("🎙️ Speech Emotion Recognition")
st.write("Record your speech or upload a WAV file.")

recording = st.audio_input("Record your voice")
uploaded = st.file_uploader(
    "Or upload a WAV file",
    type=["wav"]
)

audio = recording if recording is not None else uploaded

if audio is not None:
    st.audio(audio)

    if st.button("Predict Emotion", type="primary"):
        try:
            with st.spinner("Analyzing speech..."):
                emotion, probabilities = predict_emotion(
                    audio.getvalue()
                )

            st.subheader(f"Predicted emotion: {emotion.title()}")

            st.write("Prediction scores")
            for label, score in sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            ):
                st.write(f"{label.title()}: {score:.1%}")
                st.progress(float(score))

            st.caption(
                "These are model scores, not a guaranteed "
                "measurement of actual human emotion."
            )

        except Exception as error:
            st.error(f"Audio processing failed: {error}")
   