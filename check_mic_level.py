"""Run this while staying silent. Shows your room's ambient RMS level."""
import sys, time
import numpy as np
try:
    import sounddevice as sd
except ImportError:
    print("pip install sounddevice"); sys.exit(1)

SAMPLE_RATE = 22050
DURATION    = 4.0
SAMPLES     = int(SAMPLE_RATE * DURATION)

print("Stay completely silent for 5 readings...")
print()

levels = []
for i in range(5):
    audio = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    levels.append(rms)
    print(f"  Reading {i+1}: rms = {rms:.4f}")

print()
print(f"  Average silence rms : {np.mean(levels):.4f}")
print(f"  Max silence rms     : {np.max(levels):.4f}")
print()
print(f"  Recommended SILENCE_THRESHOLD = {np.max(levels) * 2.5:.4f}")
