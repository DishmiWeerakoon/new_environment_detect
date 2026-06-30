"""
recognizer.py
-------------
High-level integration API for the Environmental Context Recognition system.
Binary CNN classifier: Transportation (0) vs Conversation (1).

Designed for use in a hearing-impaired assistive app:
  - Single class to load once, call many times
  - Real-time microphone stream with a thread-safe callback
  - Transportation alert triggered on every transportation-mode window
  - Stable decisions via majority voting over a rolling buffer

Typical usage
-------------
    from src.recognizer import EnvironmentRecognizer

    rec = EnvironmentRecognizer()

    # --- File or array ---
    result = rec.predict_file("audio.wav")
    print(result["label"], result["confidence"])

    # --- Live microphone ---
    def on_result(result):
        if result["alert_sound"]:
            app.trigger_danger_alert()
        if result["changed"]:
            app.update_environment_display(result["label"], result["confidence"])

    rec.start_stream(callback=on_result)
    # ... app runs ...
    rec.stop_stream()
"""

import os
import threading
import queue
from collections import deque, Counter
from typing import Callable

import numpy as np

from .feature_extraction import extract_mel_from_segment, sliding_window_mel
from .cnn_classifier import CNNClassifier

ENVIRONMENT_MODES = {
    "transportation": "Transportation Mode",
    "conversation":   "Conversation Mode",
    "unknown":        "Unknown",
}

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "cnn_model.keras"
)


class EnvironmentRecognizer:
    """
    CNN-based environment recognizer for hearing-impaired users.

    Classifies every audio window as Transportation or Conversation and
    emits stable environment decisions via majority voting.

    Parameters
    ----------
    model_path  : path to cnn_model.keras
    sample_rate : audio sample rate used during training (default 22050)
    window_size : analysis window in seconds (default 4.0 — matches training)
    hop_size    : step between windows in seconds (default 2.0)
    vote_window : number of recent predictions used for majority vote (default 5)
    """

    def __init__(
        self,
        model_path:  str   = _DEFAULT_MODEL,
        sample_rate: int   = 22050,
        window_size: float = 4.0,
        hop_size:    float = 2.0,
        vote_window: int   = 5,
    ):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size    = hop_size
        self.vote_window = vote_window

        self._clf = CNNClassifier().load(os.path.normpath(model_path))

        # Stream state
        self._stream        = None
        self._stream_thread = None
        self._stop_event    = threading.Event()
        self._result_queue: queue.Queue = queue.Queue()
        self._mic_buffer    = np.array([], dtype=np.float32)
        self._last_env      = None
        self._env_vote_buf: deque = deque(maxlen=vote_window)

    # ------------------------------------------------------------------
    # Public: file / array inference
    # ------------------------------------------------------------------

    def predict_file(self, audio_path: str, duration: float = None) -> dict:
        """Run inference on a WAV / MP3 / FLAC file."""
        import librosa
        audio, sr = librosa.load(
            audio_path, sr=self.sample_rate, mono=True, duration=duration
        )
        return self.predict_array(audio, sr)

    def predict_array(self, audio: np.ndarray, sr: int = None) -> dict:
        """
        Run inference on a raw float32 numpy audio array.

        Returns
        -------
        dict with keys:
            environment   : 'transportation' | 'conversation' | 'unknown'
            label         : human-readable mode string
            confidence    : float 0-1  (fraction of windows agreeing)
            dominant_sound: most frequent window label
            sound_counts  : dict {label: count}
            alert_sound   : 'transportation' when transportation detected, else None
            breakdown     : {'transportation_ratio': ..., 'conversation_ratio': ...}
            changed       : always False for single-call inference
        """
        if sr is None:
            sr = self.sample_rate

        mel_list, _ = sliding_window_mel(
            audio, sample_rate=sr,
            window_size=self.window_size,
            hop_size=self.hop_size,
        )
        if not mel_list:
            return self._empty_result()

        X = np.array(mel_list)[..., np.newaxis]   # (N, 128, 173, 1)
        labels = self._clf.predict_labels(X)

        counts = Counter(labels)
        total  = len(labels)
        n_t    = counts.get("transportation", 0)
        n_c    = counts.get("conversation",   0)
        conf_t = n_t / total
        conf_c = n_c / total

        if n_t > n_c:
            environment, confidence = "transportation", conf_t
        elif n_c > 0:
            environment, confidence = "conversation", conf_c
        else:
            environment, confidence = "unknown", 0.0

        return {
            "environment":    environment,
            "label":          ENVIRONMENT_MODES[environment],
            "confidence":     round(confidence, 3),
            "dominant_sound": counts.most_common(1)[0][0],
            "sound_counts":   dict(counts),
            "alert_sound":    "transportation" if environment == "transportation" else None,
            "breakdown":      {"transportation_ratio": round(conf_t, 3),
                               "conversation_ratio":   round(conf_c, 3)},
            "changed":        False,
        }

    # ------------------------------------------------------------------
    # Public: live microphone stream
    # ------------------------------------------------------------------

    def start_stream(self, callback: Callable, device: int = None):
        """
        Start capturing from the microphone and call *callback* on every result.

        The callback receives the same dict as predict_array(), with the extra
        'changed' flag set to True when the environment mode changes.
        """
        if self._stream is not None:
            raise RuntimeError("Stream already running. Call stop_stream() first.")

        self._stop_event.clear()
        self._mic_buffer = np.array([], dtype=np.float32)
        self._last_env   = None
        self._env_vote_buf.clear()

        self._stream_thread = threading.Thread(
            target=self._stream_loop, args=(callback, device), daemon=True
        )
        self._stream_thread.start()

    def stop_stream(self):
        """Stop the microphone stream and wait for the thread to exit."""
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=3.0)
            self._stream_thread = None

    @property
    def is_streaming(self) -> bool:
        return self._stream_thread is not None and self._stream_thread.is_alive()

    # ------------------------------------------------------------------
    # Internal: stream loop
    # ------------------------------------------------------------------

    def _stream_loop(self, callback: Callable, device):
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone streaming. "
                "Run: pip install sounddevice"
            )

        hop_samples    = int(self.hop_size    * self.sample_rate)
        window_samples = int(self.window_size * self.sample_rate)

        def _audio_callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            self._result_queue.put(mono.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=hop_samples,
            device=device,
            callback=_audio_callback,
        )
        self._stream.start()

        while not self._stop_event.is_set():
            try:
                chunk = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._mic_buffer = np.concatenate([self._mic_buffer, chunk])

            while len(self._mic_buffer) >= window_samples:
                segment          = self._mic_buffer[:window_samples]
                self._mic_buffer = self._mic_buffer[hop_samples:]

                mel   = extract_mel_from_segment(segment, sample_rate=self.sample_rate)
                X     = mel[np.newaxis, ..., np.newaxis]   # (1, 128, 173, 1)
                label = self._clf.predict_labels(X)[0]

                self._env_vote_buf.append(label)
                voted_env  = Counter(self._env_vote_buf).most_common(1)[0][0]
                vote_conf  = Counter(self._env_vote_buf)[voted_env] / len(self._env_vote_buf)

                changed        = voted_env != self._last_env
                self._last_env = voted_env

                result = {
                    "environment":    voted_env,
                    "label":          ENVIRONMENT_MODES[voted_env],
                    "confidence":     round(vote_conf, 3),
                    "dominant_sound": label,
                    "sound_counts":   dict(Counter(self._env_vote_buf)),
                    "alert_sound":    "transportation" if label == "transportation" else None,
                    "breakdown":      {},
                    "changed":        changed,
                }

                try:
                    callback(result)
                except Exception as e:
                    print(f"[EnvironmentRecognizer] Callback error: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self) -> dict:
        return {
            "environment":    "unknown",
            "label":          ENVIRONMENT_MODES["unknown"],
            "confidence":     0.0,
            "dominant_sound": "unknown",
            "sound_counts":   {},
            "alert_sound":    None,
            "breakdown":      {},
            "changed":        False,
        }
