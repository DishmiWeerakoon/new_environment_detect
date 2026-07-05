"""
Quick test for EnvironmentRecognizer (CNN binary classifier).
Run from the project root:  python test_recognizer.py

Requires models/cnn_model.keras to exist — run notebook 04 first.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath('.'))
from src.recognizer import EnvironmentRecognizer

MODEL_PATH = "models/cnn_model_v2.keras"
if not os.path.isfile(MODEL_PATH):
    MODEL_PATH = "models/cnn_model.keras"

if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: Model not found. Run notebook 04 first.")
    sys.exit(1)

rec = EnvironmentRecognizer(model_path=MODEL_PATH)
print("CNN model loaded.\n")

# ──────────────────────────────────────────────
# 1.  FILE TESTS
# ──────────────────────────────────────────────
print("=" * 55)
print("FILE TESTS")
print("=" * 55)

test_files = [
    ("data/raw/custom_audio/conversation/ES2008a.Mix-Headset.wav", "conversation"),
    ("data/raw/transpotation/" + (
        next((f for f in os.listdir("data/raw/transpotation")
              if f.endswith(('.mp3','.wav'))), "")
        if os.path.isdir("data/raw/transpotation") else ""
    ), "transportation"),
]

for filepath, expected in test_files:
    if not filepath or not os.path.isfile(filepath):
        print(f"  Skipping (file not found): {filepath}")
        continue

    result = rec.predict_file(filepath, duration=30.0)
    status = "PASS" if result['environment'] == expected else "FAIL"

    print(f"\n[{status}] {os.path.basename(filepath)}")
    print(f"  Expected    : {expected}")
    print(f"  Predicted   : {result['label']}")
    print(f"  Confidence  : {result['confidence']:.0%}")
    print(f"  Breakdown   : transport={result['breakdown'].get('transportation_ratio', 0):.0%}"
          f"  conversation={result['breakdown'].get('conversation_ratio', 0):.0%}")
    print(f"  Alert       : {result['alert_sound'] or 'none'}")

# ──────────────────────────────────────────────
# 2.  MICROPHONE TEST  (10 seconds live)
# ──────────────────────────────────────────────
print()
print("=" * 55)
print("MICROPHONE TEST  (10 seconds — speak or make noise)")
print("=" * 55)

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Run:  pip install sounddevice")
    sys.exit(0)

results_log = []

def on_result(result):
    tag   = "[TRANSPORT] " if result["alert_sound"] else "            "
    arrow = " <-- CHANGED" if result["changed"] else ""
    print(f"  {tag}{result['label']:25s}  conf={result['confidence']:.0%}{arrow}")
    results_log.append(result)

print("Listening... (Ctrl+C to stop early)")
rec.start_stream(callback=on_result)

try:
    time.sleep(10)
except KeyboardInterrupt:
    pass

rec.stop_stream()

print()
print("── Session summary ──")
if results_log:
    from collections import Counter
    envs  = Counter(r["environment"] for r in results_log)
    total = len(results_log)
    for env, count in envs.most_common():
        label = {"transportation": "Transportation Mode",
                 "conversation": "Conversation Mode"}.get(env, env)
        print(f"  {label:30s} {count}/{total} windows ({count/total:.0%})")
else:
    print("  No results captured.")
