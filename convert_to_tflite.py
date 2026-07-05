"""
Convert cnn_model_v2.keras to TFLite format for mobile deployment.
Run: python convert_to_tflite.py
Output: models/cnn_model_v2.tflite
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

MODEL_PATH  = "models/cnn_model_v2.keras"
OUTPUT_PATH = "models/cnn_model_v2.tflite"

print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

print("\nConverting to TFLite ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # float16 quantization — smaller file
tflite_model = converter.convert()

os.makedirs("models", exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"Saved: {OUTPUT_PATH}  ({size_kb:.1f} KB)")

# Verify the TFLite model works
print("\nVerifying TFLite model ...")
interpreter = tf.lite.Interpreter(model_path=OUTPUT_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print(f"  Input  shape : {inp['shape']}  dtype: {inp['dtype']}")
print(f"  Output shape : {out['shape']}  dtype: {out['dtype']}")

# Run a test inference with a dummy mel spectrogram
dummy = np.zeros((1, 128, 173, 1), dtype=np.float32)
interpreter.set_tensor(inp['index'], dummy)
interpreter.invoke()
result = interpreter.get_tensor(out['index'])
print(f"  Test output  : {result[0][0]:.4f}  (p(conversation) for silent input)")
print("\nTFLite conversion successful.")
print(f"\nCopy this file to your Flutter project:")
print(f"  {os.path.abspath(OUTPUT_PATH)}")
print(f"  → YourFlutterApp/assets/models/cnn_model_v2.tflite")
