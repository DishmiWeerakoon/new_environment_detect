"""
app_server.py — FastAPI server for mobile integration (REST API path).

Start:   uvicorn app_server:app --host 0.0.0.0 --port 8000
Mobile app POSTs 4-second WAV audio → receives mode label JSON.

Requires:  pip install fastapi uvicorn python-multipart
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np

from src.recognizer import EnvironmentRecognizer

app = FastAPI(title="Environment Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once at startup
print("Loading CNN model...")
rec = EnvironmentRecognizer()
print("Model ready.\n")


@app.get("/health")
def health():
    return {"status": "ok", "model": "cnn_model.keras"}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """
    Accepts a WAV/MP3 audio file (ideally 4 seconds).
    Returns:
        mode        — "transportation" | "conversation"
        label       — "Transportation Mode" | "Conversation Mode"
        confidence  — 0.0 to 1.0
        alert       — true if transportation detected (trigger vibration)
    """
    if not audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    data = await audio.read()
    try:
        wav, sr = librosa.load(io.BytesIO(data), sr=22050, mono=True, duration=4.0)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

    result = rec.predict_array(wav, sr)

    return {
        "mode":       result["environment"],
        "label":      result["label"],
        "confidence": result["confidence"],
        "alert":      result["alert_sound"] is not None,
        "breakdown":  result["breakdown"],
    }
