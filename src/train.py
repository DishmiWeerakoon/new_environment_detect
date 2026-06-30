"""
train.py
--------
Trains a Random Forest or SVM sound classifier on extracted features and saves
the model + scaler to disk.

Usage
-----
    python src/train.py \
        --features data/processed/features/features.csv \
        --model    models/ \
        [--type    random_forest | svm] \
        [--cv      5]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sound_classifier import SoundClassifier, URBANSOUND_CLASSES


def load_features(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["class_id"].values.astype(int)
    return X, y, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model",    required=True)
    parser.add_argument("--type",     default="random_forest",
                        choices=["random_forest", "svm"])
    parser.add_argument("--cv",       type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.model, exist_ok=True)

    print("Loading features …")
    X, y, df = load_features(args.features)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(set(y))} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SoundClassifier(model_type=args.type).build_model()

    print(f"\nCross-validating ({args.cv}-fold) …")
    scores = clf.cross_validate(X_train, y_train, cv=args.cv)
    print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    print("\nFitting on full training set …")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    target_names = [URBANSOUND_CLASSES[i] for i in sorted(set(y))]
    print("\nTest-set classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    model_path  = os.path.join(args.model, f"{args.type}_model.pkl")
    scaler_path = os.path.join(args.model, "scaler.pkl")
    clf.save(model_path, scaler_path)


if __name__ == "__main__":
    main()
