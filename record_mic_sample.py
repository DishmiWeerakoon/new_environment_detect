"""
Record mic speech samples to add to conversation training data.
Run:  python record_mic_sample.py

Saves to: data/raw/custom_audio/conversation/mic_recording_<n>.wav
Then re-run notebooks 02 -> 03 -> 04 to retrain.
"""

import sys
import os
import time
import wave
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Run:  venv\\Scripts\\pip install sounddevice")
    sys.exit(1)

SAMPLE_RATE  = 22050
DURATION     = 60       # seconds per recording
OUTPUT_DIR   = "data/raw/custom_audio/conversation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find next free filename
n = 1
while os.path.isfile(os.path.join(OUTPUT_DIR, f"mic_recording_{n}.wav")):
    n += 1
out_path = os.path.join(OUTPUT_DIR, f"mic_recording_{n}.wav")

print("=" * 55)
print(f"Recording {DURATION}s of mic speech → {out_path}")
print("=" * 55)
print()
print("Tips for best results:")
print("  - Speak naturally, as if in a conversation")
print("  - Mix: talking, reading aloud, phone call etc.")
print("  - Keep mic distance like normal laptop use")
print()
print(f"Starting in 3 seconds... get ready to speak!")
time.sleep(3)
print("RECORDING NOW — speak continuously...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)
sd.wait()
print("Done recording.\n")

# Check RMS
rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"RMS level: {rms:.4f}", end="  ")
if rms < 0.001:
    print("(very quiet — check mic volume in Windows settings)")
elif rms < 0.01:
    print("(quiet — may want to increase mic volume)")
else:
    print("(good level)")

# Save as WAV
audio_int16 = (audio * 32767).astype(np.int16)
with wave.open(out_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_int16.tobytes())

size_mb = os.path.getsize(out_path) / 1e6
print(f"Saved: {out_path}  ({size_mb:.1f} MB)")
print()
print("Next steps:")
print("  1. Run this script again to record more samples (aim for 3-5 recordings)")
print("  2. Re-run notebooks: 02 -> 03 -> 04")
print("  3. Test again with:  python test_recognizer.py")
