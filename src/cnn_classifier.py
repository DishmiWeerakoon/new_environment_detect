import numpy as np
import os

LABEL_MAP = {0: "transportation", 1: "conversation"}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# Expected mel spectrogram dimensions (must match feature_extraction defaults)
# sr=22050, duration=4.0s, n_mels=128, hop_length=512 → 173 time frames
CNN_INPUT_SHAPE = (128, 173, 1)


class CNNClassifier:
    """
    CNN-based binary audio classifier: Transportation (0) vs Conversation (1).

    Input: (N, 128, 173, 1)  — batch of normalized mel spectrograms
    Output: binary label 0 or 1 per sample
    """

    def __init__(self):
        self.model = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, input_shape=CNN_INPUT_SHAPE):
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Input(shape=input_shape),

            # Block 1: 32 filters
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2: 64 filters
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3: 128 filters
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classifier head
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return self

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=50, batch_size=32, class_weight=None):
        from tensorflow import keras

        if X_train.ndim == 3:
            X_train = X_train[..., np.newaxis]
        if X_val is not None and X_val.ndim == 3:
            X_val = X_val[..., np.newaxis]

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6, monitor="val_loss", verbose=1
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        if X.ndim == 3:
            X = X[..., np.newaxis]
        proba = self.model.predict(X, verbose=0).flatten()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Return (N, 2) array: [:, 0] = P(transportation), [:, 1] = P(conversation)."""
        if X.ndim == 3:
            X = X[..., np.newaxis]
        p_conv = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - p_conv, p_conv])

    def predict_labels(self, X) -> list:
        return [LABEL_MAP[int(p)] for p in self.predict(X)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str):
        self.model.save(model_path)
        print(f"Saved CNN model → {model_path}")

    def load(self, model_path: str):
        from tensorflow import keras
        self.model = keras.models.load_model(os.path.normpath(model_path))
        return self
