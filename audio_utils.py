# audio_utils.py

import numpy as np
import librosa

def load_audio(file_path: str) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=None)
    return audio

def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    # Example preprocessing: normalize audio
    return audio / np.max(np.abs(audio))