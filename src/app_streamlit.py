import streamlit as st
import pandas as pd
from feature_extraction import extract_features
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import time

# Configure page
st.set_page_config(
    page_title="Music Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalistic CSS with dark mode support and 20% size reduction
st.markdown("""
<style>
    footer, header {display: none;}
    .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem;}
    
    /* Light mode variables */
    :root {
        --model-card-bg: #f0f2f5;
        --model-card-border: #d9d9d9;
        --model-card-text: #333333;
        --upload-section-bg: #fafafa;
        --upload-section-hover-bg: #ffffff;
        --upload-section-border: #d9d9d9;
        --upload-section-hover-border: #2c3e50;
        --result-card-bg: #f0f2f5;
        --result-card-border: #d9d9d9;
        --result-card-text: #333333;
        --prediction-label-bg: #2c3e50;
        --prediction-label-text: white;
        --section-header-color: #2c3e50;
        --heading-color: #2c3e50;
        --subtitle-color: #555;
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --model-card-bg: rgba(49, 51, 63, 0.4);
            --model-card-border: rgba(250, 250, 250, 0.2);
            --model-card-text: rgba(250, 250, 250, 0.9);
            --upload-section-bg: rgba(49, 51, 63, 0.3);
            --upload-section-hover-bg: rgba(49, 51, 63, 0.5);
            --upload-section-border: rgba(250, 250, 250, 0.2);
            --upload-section-hover-border: rgba(255, 75, 75, 0.6);
            --result-card-bg: rgba(49, 51, 63, 0.4);
            --result-card-border: rgba(250, 250, 250, 0.2);
            --result-card-text: rgba(250, 250, 250, 0.9);
            --prediction-label-bg: rgba(255, 75, 75, 0.8);
            --prediction-label-text: white;
            --section-header-color: rgba(250, 250, 250, 0.9);
            --heading-color: rgba(250, 250, 250, 0.95);
            --subtitle-color: rgba(250, 250, 250, 0.7);
        }
    }
    
    /* Streamlit dark mode class override */
    [data-testid="stAppViewContainer"][data-theme="dark"] {
        --model-card-bg: rgba(49, 51, 63, 0.4);
        --model-card-border: rgba(250, 250, 250, 0.2);
        --model-card-text: rgba(250, 250, 250, 0.9);
        --upload-section-bg: rgba(49, 51, 63, 0.3);
        --upload-section-hover-bg: rgba(49, 51, 63, 0.5);
        --upload-section-border: rgba(250, 250, 250, 0.2);
        --upload-section-hover-border: rgba(255, 75, 75, 0.6);
        --result-card-bg: rgba(49, 51, 63, 0.4);
        --result-card-border: rgba(250, 250, 250, 0.2);
        --result-card-text: rgba(250, 250, 250, 0.9);
        --prediction-label-bg: rgba(255, 75, 75, 0.8);
        --prediction-label-text: white;
        --section-header-color: rgba(250, 250, 250, 0.9);
        --heading-color: rgba(250, 250, 250, 0.95);
        --subtitle-color: rgba(250, 250, 250, 0.7);
    }
    
    .model-card {
        background: var(--model-card-bg);
        color: var(--model-card-text);
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid var(--model-card-border);
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
    }
    .model-card:hover {
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    .model-card h4 {
        color: var(--model-card-text);
        margin: 0 0 0.5rem 0;
    }
    .model-card p {
        color: var(--model-card-text);
        opacity: 0.9;
        margin: 0;
    }
    
    .upload-section {
        background: var(--upload-section-bg);
        padding: 1.6rem;
        border-radius: 8px;
        border: 2px dashed var(--upload-section-border);
        text-align: center;
        margin-bottom: 1.2rem;
        transition: all 0.2s ease;
    }
    .upload-section:hover {
        border-color: var(--upload-section-hover-border);
        background: var(--upload-section-hover-bg);
    }
    .upload-section h4 {
        color: var(--section-header-color);
        margin: 0;
    }
    
    .result-card {
        background: var(--result-card-bg);
        color: var(--result-card-text);
        padding: 1.6rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--result-card-border);
        margin-top: 1.2rem;
    }
    .result-card h2 {
        color: var(--result-card-text);
        margin: 0 0 1rem 0;
    }
    .result-card p {
        color: var(--result-card-text);
        opacity: 0.9;
    }
    
    .prediction-label {
        background: var(--prediction-label-bg);
        color: var(--prediction-label-text);
        padding: 0.64rem 1.6rem;
        border-radius: 5px;
        font-size: 0.96rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.8rem 0;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .section-header {
        color: var(--section-header-color);
        font-size: 0.96rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }
    
    .main-title {
        color: var(--heading-color);
    }
    
    .main-subtitle {
        color: var(--subtitle-color);
    }
</style>
""", unsafe_allow_html=True)

# Minimal title
st.markdown("""
<h1 class="main-title" style="font-size:3rem; font-weight:500; margin-bottom:0.8rem;">
Music Analyzer
</h1>
<p class="main-subtitle" style="margin-top:0; margin-bottom:0.8rem;">AI-Powered Music Genre Classification</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### How It Works")
    st.markdown("1. Choose your AI model\n2. Upload an audio file\n3. Get instant predictions")
    st.markdown("---")
    st.markdown("### Model Information")
    model_info = st.expander("Learn about the models", expanded=False)
    with model_info:
        st.write("**Sklearn Model**: Uses extracted audio features like tempo, pitch, and spectral characteristics.")
        st.write("**CNN Model**: Analyzes mel-spectrograms using deep learning for visual pattern recognition.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">Select AI Model</div>', unsafe_allow_html=True)
    
    # Paths
    SKLEARN_MODEL_PATH = Path("models/sklearn_model.pkl")
    CNN_MODEL_PATH = Path("models/cnn_model.pth")
    
    model_choice = st.radio("Choose your preferred model:", ["Sklearn (features)", "CNN (spectrograms)"])
    
    clf, label_encoder, scaler, cnn_model, classes = None, None, None, None, None
    
    with st.spinner("Loading AI model..."):
        if model_choice == "Sklearn (features)" and SKLEARN_MODEL_PATH.exists():
            obj = joblib.load(SKLEARN_MODEL_PATH)
            clf = obj["model"]
            scaler = obj["scaler"]
            label_encoder = obj["label_encoder"]
            st.success("Sklearn model loaded")
            with col2:
                st.markdown("""
                <div class="model-card">
                    <h4>Feature-Based Model</h4>
                    <p>Analyzing audio characteristics like tempo, pitch, and spectral features.</p>
                </div>
                """, unsafe_allow_html=True)
        elif model_choice == "CNN (spectrograms)" and CNN_MODEL_PATH.exists():
            checkpoint = torch.load(CNN_MODEL_PATH, map_location="cpu")
            classes = checkpoint["classes"]
            num_classes = len(classes)
            cnn_model = models.resnet18(weights=None)
            cnn_model.fc = nn.Linear(cnn_model.fc.in_features, num_classes)
            cnn_model.load_state_dict(checkpoint["state_dict"])
            cnn_model.eval()
            st.success("CNN model loaded")
            with col2:
                st.markdown("""
                <div class="model-card">
                    <h4>Deep Learning Model</h4>
                    <p>Using convolutional neural networks to analyze visual patterns in audio spectrograms.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Selected model not found!")
            st.stop()

# File upload
st.markdown('<div class="section-header">Upload Audio File</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop your audio file here or click to browse", type=['wav', 'mp3'])

if uploaded:
    audio_col1, audio_col2 = st.columns([1, 1])
    with audio_col1:
        st.markdown('<div class="upload-section"><h4>Audio Preview</h4></div>', unsafe_allow_html=True)
        tmp_file = Path("tmp_upload.wav")
        with tmp_file.open("wb") as f: f.write(uploaded.getbuffer())
        st.audio(str(tmp_file))
        st.info(f"**File**: {uploaded.name} | **Size**: {len(uploaded.getvalue())/1024:.1f} KB")
    
    with audio_col2:
        st.markdown('<div class="section-header">Processing Status</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if model_choice == "Sklearn (features)":
            status_text.text("Extracting audio features...")
            progress_bar.progress(25)
            time.sleep(0.5)
            feats = extract_features(str(tmp_file))
            progress_bar.progress(50)
            status_text.text("Scaling features...")
            feats_scaled = scaler.transform([feats])
            progress_bar.progress(75)
            status_text.text("Making prediction...")
            pred = clf.predict(feats_scaled)[0]
            pred_label = label_encoder.inverse_transform([pred])[0]
            progress_bar.progress(100)
            status_text.text("Done")
        
        elif model_choice == "CNN (spectrograms)":
            status_text.text("Generating spectrogram...")
            progress_bar.progress(20)
            spec_path = "tmp_spec.png"
            extract_features(str(tmp_file), save_melspec_path=spec_path)
            progress_bar.progress(40)
            status_text.text("Processing with CNN...")
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            image = Image.open(spec_path).convert("RGB")
            image = transform(image).unsqueeze(0)
            progress_bar.progress(70)
            status_text.text("Making prediction...")
            with torch.no_grad():
                outputs = cnn_model(image)
                _, pred_idx = torch.max(outputs, 1)
                pred_label = classes[pred_idx.item()]
            progress_bar.progress(100)
            status_text.text("Done")
    
    # Results
    st.markdown(f"""
    <div class="result-card">
        <h2>🎯 Prediction Results</h2>
        <div class="prediction-label">{pred_label}</div>
        <p>Genre classified using {model_choice} model</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Analysis Insights")
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    insight_col1.metric("🎵 Predicted Genre", pred_label)
    insight_col2.metric("🤖 Model Used", model_choice.split()[0])
    insight_col3.metric("📁 File Format", uploaded.name.split('.')[-1].upper())