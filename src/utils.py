import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(file_path: str, sample_rate: int = 22050, duration: float | None = None):
    if file_path.lower().endswith('.m4a'):
        from src.feature_extraction import _load_audio
        audio, sr = _load_audio(file_path, sample_rate=sample_rate, duration=duration)
        return audio, sr
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    return audio, sr


def generate_sliding_windows(audio: np.ndarray, sr: int,
                              window_size: float = 2.0, hop_size: float = 1.0):
    """Yield (segment, start_time_seconds) pairs for each window."""
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    windows, timestamps = [], []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        windows.append(audio[start: start + window_samples])
        timestamps.append(start / sr)
    return windows, timestamps


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_waveform(audio: np.ndarray, sr: int, title: str = "Waveform"):
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_mel_spectrogram(audio: np.ndarray, sr: int, title: str = "Mel Spectrogram"):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 40, title: str = "MFCC"):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          title: str = "Confusion Matrix"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def plot_environment_timeline(timestamps: list[float], environments: list[str],
                               confidences: list[float],
                               title: str = "Environment Detection Timeline"):
    color_map = {"transportation": "tomato", "conversation": "mediumseagreen", "unknown": "silver"}
    plt.figure(figsize=(14, 4))
    for t, env, conf in zip(timestamps, environments, confidences):
        plt.bar(t, conf, width=0.8, color=color_map.get(env, "gray"), alpha=0.8)
    legend_handles = [
        mpatches.Patch(color="tomato",         label="Transportation"),
        mpatches.Patch(color="mediumseagreen",  label="Conversation"),
        mpatches.Patch(color="silver",          label="Unknown"),
    ]
    plt.legend(handles=legend_handles)
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# UrbanSound8K helpers
# ---------------------------------------------------------------------------

def get_urbansound_metadata(dataset_path: str):
    import pandas as pd
    csv_path = os.path.join(dataset_path, "metadata", "UrbanSound8K.csv")
    return pd.read_csv(csv_path)


def get_audio_file_path(dataset_path: str, fold: int, filename: str) -> str:
    return os.path.join(dataset_path, "audio", f"fold{fold}", filename)
