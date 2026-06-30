import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# UrbanSound8K native classes (IDs 0-9) + custom speech/conversation (10-11)
URBANSOUND_CLASSES = {
    0:  "air_conditioner",
    1:  "car_horn",
    2:  "children_playing",
    3:  "dog_bark",
    4:  "drilling",
    5:  "engine_idling",
    6:  "gun_shot",
    7:  "jackhammer",
    8:  "siren",
    9:  "street_music",
    10: "speech",
    11: "conversation",
}

CLASS_TO_ID = {v: k for k, v in URBANSOUND_CLASSES.items()}

TRANSPORTATION_SOUNDS = {
    "car_horn", "engine_idling", "siren",
    "jackhammer", "drilling", "air_conditioner",
}
CONVERSATION_SOUNDS = {"speech", "conversation", "children_playing"}
AMBIENT_SOUNDS = {"dog_bark", "street_music", "gun_shot"}


class SoundClassifier:
    """Thin wrapper around Random Forest or SVM with a bundled StandardScaler."""

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in ("random_forest", "svm"):
            raise ValueError("model_type must be 'random_forest' or 'svm'")
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_model(self, **kwargs):
        """Instantiate the underlying estimator. Call before fit()."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 10),
                gamma=kwargs.get("gamma", "scale"),
                probability=True,
                random_state=42,
            )
        return self

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5):
        """Return per-fold accuracy scores (scaler is refitted inside each fold)."""
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", self.model),
        ])
        return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def predict_labels(self, X: np.ndarray) -> list[str]:
        """Return human-readable class names instead of integer IDs."""
        return [URBANSOUND_CLASSES.get(int(p), "unknown") for p in self.predict(X)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str, scaler_path: str):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved model  → {model_path}")
        print(f"Saved scaler → {scaler_path}")

    def load(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        return self

    # ------------------------------------------------------------------
    # Introspection (Random Forest only)
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self):
        if self.model_type != "random_forest":
            raise AttributeError("feature_importances_ only available for Random Forest")
        return self.model.feature_importances_

    @property
    def classes_(self):
        return self.model.classes_
