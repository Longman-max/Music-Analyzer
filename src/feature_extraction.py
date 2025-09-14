import librosa
import numpy as np
import os
import soundfile as sf
from pathlib import Path


def load_audio(path, sr=22050, mono=True, duration=None):
    y, sr = librosa.load(path, sr=sr, mono=mono, duration=duration)
    return y, sr


def extract_features(path, sr=22050, n_mfcc=20, save_melspec_path=None):
    """Return a 1D feature vector (concatenated summary statistics) and optionally save a mel-spectrogram image.

    Features included:
    - MFCC (mean, std)
    - Chroma (mean, std)
    - Spectral Contrast (mean, std)
    - Tonnetz (mean, std)
    - RMS energy (mean, std)
    - Spectral centroid (mean, std)
    - Zero-crossing rate (mean, std)
    """
    y, sr = load_audio(path, sr=sr)

    # Ensure signal is long enough
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # Tonnetz (requires harmonic)
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = np.mean(spec_centroid)
    sc_std = np.std(spec_centroid)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Concatenate all features into a 1D array
    feat = np.concatenate([
        mfcc_mean, mfcc_std,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std,
        tonnetz_mean, tonnetz_std,
        [rms_mean, rms_std, sc_mean, sc_std, zcr_mean, zcr_std]
    ])

    # Optionally save mel-spectrogram as an image (for CNN)
    if save_melspec_path:
        import matplotlib.pyplot as plt
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_melspec_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return feat