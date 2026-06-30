from collections import Counter, deque

TRANSPORTATION_SOUNDS = {
    "car_horn", "engine_idling", "siren",
    "jackhammer", "drilling", "air_conditioner",
}
CONVERSATION_SOUNDS = {"speech", "conversation", "children_playing"}
AMBIENT_SOUNDS = {"dog_bark", "street_music", "gun_shot"}

ENVIRONMENT_MODES = {
    "transportation": "Transportation Mode",
    "conversation":   "Conversation Mode",
    "unknown":        "Unknown / Ambient Mode",
}


class EnvironmentMapper:
    """
    Converts a stream of sound-event labels into a high-level environment mode
    using a sliding prediction buffer and configurable thresholds.

    Decision rule
    -------------
    Let T = fraction of buffer entries that are transportation sounds,
        C = fraction that are conversation sounds.

    - T >= transportation_threshold AND T > C  →  "transportation"
    - C >= conversation_threshold  AND C > T   →  "conversation"
    - otherwise                                →  "unknown"
    """

    def __init__(
        self,
        window_size: int = 10,
        transportation_threshold: float = 0.40,
        conversation_threshold: float = 0.35,
    ):
        self.window_size = window_size
        self.transportation_threshold = transportation_threshold
        self.conversation_threshold = conversation_threshold
        self._buffer: deque[str] = deque(maxlen=window_size)
        self.environment_history: list[dict] = []

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def add_prediction(self, sound_label: str):
        self._buffer.append(sound_label)

    def add_predictions(self, sound_labels: list[str]):
        for label in sound_labels:
            self._buffer.append(label)

    def reset(self):
        self._buffer.clear()
        self.environment_history.clear()

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def get_current_environment(self) -> tuple[str, float, dict]:
        """
        Classify the current buffer contents.

        Returns
        -------
        environment : str    ("transportation", "conversation", "unknown")
        confidence  : float  (dominant category's ratio)
        breakdown   : dict   with ratios and raw counts
        """
        if not self._buffer:
            return "unknown", 0.0, {}

        counts = Counter(self._buffer)
        total = len(self._buffer)

        t_count = sum(counts[s] for s in TRANSPORTATION_SOUNDS if s in counts)
        c_count = sum(counts[s] for s in CONVERSATION_SOUNDS if s in counts)
        t_ratio = t_count / total
        c_ratio = c_count / total

        breakdown = {
            "transportation_ratio": round(t_ratio, 3),
            "conversation_ratio":   round(c_ratio, 3),
            "ambient_ratio":        round(1 - t_ratio - c_ratio, 3),
            "counts":               dict(counts),
        }

        if t_ratio >= self.transportation_threshold and t_ratio > c_ratio:
            environment, confidence = "transportation", t_ratio
        elif c_ratio >= self.conversation_threshold and c_ratio > t_ratio:
            environment, confidence = "conversation", c_ratio
        else:
            environment, confidence = "unknown", max(t_ratio, c_ratio)

        self.environment_history.append({
            "environment": environment,
            "confidence":  round(confidence, 3),
            "breakdown":   breakdown,
        })
        return environment, round(confidence, 3), breakdown

    # ------------------------------------------------------------------
    # Majority voting over a longer audio stretch
    # ------------------------------------------------------------------

    def majority_vote(
        self,
        sound_labels: list[str],
        window_size: int | None = None,
        hop: int = 1,
    ) -> tuple[str, float]:
        """
        Slide a sub-window over *sound_labels*, classify each sub-window,
        then return the most common environment across all sub-windows.

        Parameters
        ----------
        sound_labels : list of per-frame sound predictions
        window_size  : sub-window size in frames (defaults to self.window_size)
        hop          : stride in frames
        """
        ws = window_size or self.window_size
        environments: list[str] = []

        for start in range(0, len(sound_labels) - ws + 1, hop):
            sub = sound_labels[start: start + ws]
            tmp = EnvironmentMapper(
                window_size=ws,
                transportation_threshold=self.transportation_threshold,
                conversation_threshold=self.conversation_threshold,
            )
            tmp.add_predictions(sub)
            env, _, _ = tmp.get_current_environment()
            environments.append(env)

        if not environments:
            return "unknown", 0.0

        counts = Counter(environments)
        winner, win_count = counts.most_common(1)[0]
        return winner, round(win_count / len(environments), 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def label(environment_key: str) -> str:
        """Human-readable label for an environment key."""
        return ENVIRONMENT_MODES.get(environment_key, "Unknown Mode")

    def summarize(self) -> dict:
        """Aggregate statistics over the full environment history."""
        if not self.environment_history:
            return {}
        envs = [e["environment"] for e in self.environment_history]
        counts = Counter(envs)
        total = len(envs)
        return {k: round(v / total, 3) for k, v in counts.items()}
