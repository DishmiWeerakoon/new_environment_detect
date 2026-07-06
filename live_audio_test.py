"""
Live microphone test for the final CNN environment model.

Run from the project root:
    python live_audio_test.py

The model expects 4-second audio windows. This script keeps a rolling mic
buffer and updates the displayed prediction every hop interval.
"""

import argparse
import os
import queue
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.cnn_classifier import CNNClassifier, CONVERSATION_THRESHOLD
from src.feature_extraction import extract_mel_from_segment


DEFAULT_MODEL = "models/cnn_model_v2.keras"
FALLBACK_MODEL = "models/cnn_model.keras"


def resolve_model_path(model_path: str | None) -> str:
    if model_path:
        return model_path
    if os.path.isfile(DEFAULT_MODEL):
        return DEFAULT_MODEL
    return FALLBACK_MODEL


def format_prediction(label: str, confidence: float, p_conversation: float, rms: float) -> str:
    return (
        f"Mode: {label:<19} "
        f"Confidence: {confidence:6.1%}  "
        f"P(conversation): {p_conversation:6.1%}  "
        f"RMS: {rms:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run live mic inference with the CNN model.")
    parser.add_argument("--model", default=None, help="Path to .keras model file")
    parser.add_argument("--device", type=int, default=None, help="Optional sounddevice input device id")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--window-size", type=float, default=4.0, help="Seconds per model window")
    parser.add_argument("--hop-size", type=float, default=1.0, help="Seconds between predictions")
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.0,
        help="If > 0, windows below this RMS are shown as silence instead of classified",
    )
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed. Run: pip install sounddevice")
        return 1

    model_path = resolve_model_path(args.model)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return 1

    print(f"Loading model: {model_path}")
    classifier = CNNClassifier().load(model_path)
    print("Model loaded.")
    print()
    print("Listening to microphone. Press Ctrl+C to stop.")
    print(f"Window: {args.window_size:.1f}s | Update every: {args.hop_size:.1f}s")
    print()

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    sample_rate = args.sample_rate
    window_samples = int(args.window_size * sample_rate)
    hop_samples = int(args.hop_size * sample_rate)
    mic_buffer = np.array([], dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}", file=sys.stderr)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio_queue.put(mono.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=hop_samples,
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                chunk = audio_queue.get()
                mic_buffer = np.concatenate([mic_buffer, chunk])

                while len(mic_buffer) >= window_samples:
                    segment = mic_buffer[:window_samples]
                    mic_buffer = mic_buffer[hop_samples:]

                    rms = float(np.sqrt(np.mean(segment ** 2)))
                    if args.silence_threshold > 0 and rms < args.silence_threshold:
                        line = f"Mode: silence/waiting     Confidence:   n/a   RMS: {rms:.4f}"
                    else:
                        mel = extract_mel_from_segment(
                            segment,
                            sample_rate=sample_rate,
                            duration=args.window_size,
                        )
                        X = mel[np.newaxis, ..., np.newaxis]
                        p_conversation = float(classifier.predict_proba(X)[0, 1])

                        if p_conversation >= CONVERSATION_THRESHOLD:
                            label = "Conversation Mode"
                            confidence = p_conversation
                        else:
                            label = "Transportation Mode"
                            confidence = 1.0 - p_conversation

                        line = format_prediction(label, confidence, p_conversation, rms)

                    sys.stdout.write("\r" + line + " " * 8)
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
