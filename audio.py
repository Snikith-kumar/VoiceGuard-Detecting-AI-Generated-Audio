import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepfake_audio_detector.h5", compile=False)
    return model

model = load_model()

# Load audio from uploaded file
def load_audio(uploaded_file):
    try:
        audio_bytes = uploaded_file.getvalue()
        audio, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

# Extract fixed-size MFCC features
def extract_mfcc(audio, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512, fixed_length=200):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        if mfcc.shape[1] < fixed_length:
            pad_width = fixed_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :fixed_length]
        return mfcc
    except Exception as e:
        st.error(f"Error extracting MFCC features: {e}")
        return None

# Plot MFCC features
def plot_mfcc(mfcc, sr, hop_length=512):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, cmap="viridis")
    ax.set_title("MFCC Features", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("MFCC Coefficients", fontsize=12)
    plt.colorbar(format='%+2.0f dB')
    fig.tight_layout()
    return fig

# Predict using model
def predict_audio(mfcc):
    try:
        mfcc = mfcc[..., np.newaxis]  # (13, 200, 1)
        mfcc = np.expand_dims(mfcc, axis=0)  # (1, 13, 200, 1)
        prediction = model.predict(mfcc)
        label = "Real Audio" if prediction[0][0] > 0.5 else "Fake Audio"
        return label, prediction[0][0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Custom styling
def add_custom_css():
    st.markdown("""
    <style>
    .real-result {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    }
    .fake-result {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# App layout
def main():
    add_custom_css()

    st.title("🎵 VoiceGuard: Deepfake Audio Detector")
    st.markdown("Upload an audio file (WAV, MP3, FLAC) to detect if it's **Real** or **AI-Generated**.")

    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        with st.spinner("🔍 Analyzing audio..."):
            audio, sr = load_audio(uploaded_file)

            if audio is not None:
                mfcc = extract_mfcc(audio, sr)

                if mfcc is not None:
                    # MFCC display
                    st.subheader("🎛️ MFCC Visualization")
                    fig = plot_mfcc(mfcc, sr)
                    st.pyplot(fig)

                    # Prediction
                    st.subheader("🧠 Prediction")
                    label, prob = predict_audio(mfcc)

                    if label is not None:
                        if "Real" in label:
                            st.markdown(f"<div class='real-result'>{label}</div>", unsafe_allow_html=True)
                            st.progress(float(prob))
                            st.markdown(f"Confidence: {prob:.2f}")
                        else:
                            st.markdown(f"<div class='fake-result'>{label}</div>", unsafe_allow_html=True)
                            st.progress(float(1 - prob))
                            st.markdown(f"Confidence: {1 - prob:.2f}")

# Run app
if __name__ == "__main__":
    main()
