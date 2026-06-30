"""
predict.py
----------
Runs the full inference pipeline on a single audio file:
  1. Sliding-window feature extraction
  2. Sound-event classification
  3. Rule-based environment mapping with majority voting

Usage
-----
    python src/predict.py \
        --audio  test_audio/sample.wav \
        --model  models/ \
        [--type  random_forest | svm] \
        [--window_size 2.0] \
        [--hop         1.0] \
        [--buffer      10]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_extraction import sliding_window_features
from src.sound_classifier import SoundClassifier, URBANSOUND_CLASSES
from src.environment_mapper import EnvironmentMapper, ENVIRONMENT_MODES
from src.utils import load_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",       required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--type",        default="random_forest",
                        choices=["random_forest", "svm"])
    parser.add_argument("--window_size", type=float, default=2.0)
    parser.add_argument("--hop",         type=float, default=1.0)
    parser.add_argument("--buffer",      type=int,   default=10)
    args = parser.parse_args()

    model_path  = os.path.join(args.model, f"{args.type}_model.pkl")
    scaler_path = os.path.join(args.model, "scaler.pkl")

    print(f"Loading model from {model_path} …")
    clf = SoundClassifier(model_type=args.type).load(model_path, scaler_path)

    print(f"Loading audio: {args.audio}")
    audio, sr = load_audio(args.audio)
    print(f"  Duration: {len(audio)/sr:.1f}s  |  Sample rate: {sr} Hz")

    print("Extracting sliding-window features …")
    feature_list, timestamps = sliding_window_features(
        audio, sample_rate=sr,
        window_size=args.window_size,
        hop_size=args.hop,
    )

    if not feature_list:
        print("Audio too short for even one window. Exiting.")
        return

    X = np.array(feature_list)
    labels = clf.predict_labels(X)

    print("\nPer-window sound predictions:")
    print(f"  {'Time (s)':<10} {'Sound event'}")
    print(f"  {'-'*9:<10} {'-'*20}")
    for t, lbl in zip(timestamps, labels):
        print(f"  {t:<10.1f} {lbl}")

    mapper = EnvironmentMapper(
        window_size=args.buffer,
        transportation_threshold=0.40,
        conversation_threshold=0.35,
    )
    mapper.add_predictions(labels)
    env, confidence, breakdown = mapper.get_current_environment()

    final_env, vote_conf = mapper.majority_vote(labels)

    print("\n" + "=" * 50)
    print("ENVIRONMENT DETECTION RESULT")
    print("=" * 50)
    print(f"  Instantaneous:  {ENVIRONMENT_MODES[env]}  (confidence {confidence:.0%})")
    print(f"  Majority vote:  {ENVIRONMENT_MODES[final_env]}  (agreement {vote_conf:.0%})")
    print(f"\n  Sound breakdown:")
    print(f"    Transportation: {breakdown.get('transportation_ratio', 0):.0%}")
    print(f"    Conversation:   {breakdown.get('conversation_ratio', 0):.0%}")
    print(f"    Ambient/Other:  {breakdown.get('ambient_ratio', 0):.0%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
