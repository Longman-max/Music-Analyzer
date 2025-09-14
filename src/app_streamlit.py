import streamlit as st
import pandas as pd
from feature_extraction import extract_features
import joblib

st.title('Music-Analyzer — demo')

uploaded = st.file_uploader('Upload an audio file', type=['wav','mp3'])
if uploaded:
    with open('tmp_upload.wav','wb') as f:
        f.write(uploaded.getbuffer())
    st.audio('tmp_upload.wav')
    feats = extract_features('tmp_upload.wav')
    st.write('Feature vector length:', len(feats))

    # load model if exists
    try:
        clf = joblib.load('models/rf_model.joblib')
        pred = clf.predict([feats])[0]
        st.success(f'Predicted label: {pred}')
    except Exception:
        st.info('No model found. Train a model and place it in models/')