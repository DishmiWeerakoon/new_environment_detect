"""
preprocess.py
-------------
Reads UrbanSound8K (and optional custom speech clips) from disk, extracts
audio features for every clip, and saves the result to a CSV.

Usage
-----
    python src/preprocess.py \
        --dataset  data/raw/UrbanSound8K \
        --output   data/processed \
        [--custom  data/raw/custom_audio] \
        [--sr 22050] \
        [--duration 4.0] \
        [--n_mfcc 40]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_extraction import extract_features, extract_segments_from_long_file
from src.utils import get_urbansound_metadata, get_audio_file_path

CUSTOM_CLASS_MAP = {
    "speech":       10,
    "conversation": 11,
}


def process_urbansound(dataset_path: str, sr: int, duration: float,
                        n_mfcc: int) -> pd.DataFrame:
    meta = get_urbansound_metadata(dataset_path)
    rows = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="UrbanSound8K"):
        fp = get_audio_file_path(dataset_path, int(row["fold"]), row["slice_file_name"])
        if not os.path.isfile(fp):
            continue
        feats = extract_features(fp, sample_rate=sr, duration=duration, n_mfcc=n_mfcc)
        if feats is None:
            continue
        rows.append({
            "file":     fp,
            "class_id": int(row["classID"]),
            "class":    row["class"],
            "fold":     int(row["fold"]),
            **{f"f{i}": v for i, v in enumerate(feats)},
        })
    return pd.DataFrame(rows)


def process_custom(custom_path: str, sr: int, duration: float,
                   n_mfcc: int) -> pd.DataFrame:
    """
    Process custom audio files.  Long recordings (> 2× segment_duration) are
    sliced into non-overlapping chunks so that a single meeting WAV contributes
    many training samples rather than just one.
    """
    rows = []
    for label_dir in os.listdir(custom_path):
        class_id = CUSTOM_CLASS_MAP.get(label_dir.lower())
        if class_id is None:
            continue
        dir_path = os.path.join(custom_path, label_dir)
        for fname in tqdm(os.listdir(dir_path), desc=f"Custom/{label_dir}"):
            if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                continue
            fp = os.path.join(dir_path, fname)
            file_size = os.path.getsize(fp)
            # Treat files larger than ~1 MB as long recordings worth slicing
            long_recording = file_size > 1_000_000

            if long_recording:
                feat_list, n_total = extract_segments_from_long_file(
                    fp, sample_rate=sr, segment_duration=duration, n_mfcc=n_mfcc
                )
                print(f"  {fname}: {len(feat_list)}/{n_total} segments kept")
                for seg_idx, feats in enumerate(feat_list):
                    rows.append({
                        "file":     f"{fp}::seg{seg_idx}",
                        "class_id": class_id,
                        "class":    label_dir.lower(),
                        "fold":     0,
                        **{f"f{i}": v for i, v in enumerate(feats)},
                    })
            else:
                feats = extract_features(fp, sample_rate=sr, duration=duration, n_mfcc=n_mfcc)
                if feats is None:
                    continue
                rows.append({
                    "file":     fp,
                    "class_id": class_id,
                    "class":    label_dir.lower(),
                    "fold":     0,
                    **{f"f{i}": v for i, v in enumerate(feats)},
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--custom",   default=None)
    parser.add_argument("--sr",       type=int,   default=22050)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n_mfcc",   type=int,   default=40)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "features"), exist_ok=True)

    dfs = [process_urbansound(args.dataset, args.sr, args.duration, args.n_mfcc)]

    if args.custom and os.path.isdir(args.custom):
        dfs.append(process_custom(args.custom, args.sr, args.duration, args.n_mfcc))

    df = pd.concat(dfs, ignore_index=True)
    out_csv = os.path.join(args.output, "features", "features.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} samples → {out_csv}")
    print(f"Class distribution:\n{df['class'].value_counts()}")


if __name__ == "__main__":
    main()
