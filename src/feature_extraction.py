import numpy as np
import librosa


def extract_features(file_path, sample_rate=22050, duration=4.0, n_mfcc=40):
    """
    Load an audio file and return a flat feature vector.

    Features: MFCCs (mean+std), chroma (mean+std), mel spectrogram (mean+std),
    spectral centroid, spectral rolloff, zero crossing rate.
    Total dimensionality: 80 + 24 + 256 + 6 = 366
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)

        expected = int(sample_rate * duration)
        if len(audio) < expected:
            audio = np.pad(audio, (0, expected - len(audio)))

        return _compute_features(audio, sr, n_mfcc)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_segments_from_long_file(file_path, sample_rate=22050, segment_duration=4.0,
                                    n_mfcc=40, min_rms=0.001):
    """
    Slice a long audio file into non-overlapping *segment_duration*-second chunks
    and extract one feature vector per chunk.

    Chunks below *min_rms* (near-silence) are skipped so silent gaps in meeting
    recordings don't pollute the training set.

    Returns
    -------
    feature_list : list of np.ndarray  — one per valid segment
    n_segments   : int                 — total segments before silence filter
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], 0

    seg_samples = int(sr * segment_duration)
    total_segments = len(audio) // seg_samples
    feature_list = []

    for i in range(total_segments):
        segment = audio[i * seg_samples: (i + 1) * seg_samples]
        # Skip near-silent segments
        if np.sqrt(np.mean(segment ** 2)) < min_rms:
            continue
        feats = _compute_features(segment, sr, n_mfcc)
        feature_list.append(feats)

    return feature_list, total_segments


def extract_features_from_segment(audio_segment, sample_rate=22050, n_mfcc=40):
    """Return a feature vector for an already-loaded audio numpy array."""
    return _compute_features(audio_segment, sample_rate, n_mfcc)


def sliding_window_features(audio, sample_rate=22050, window_size=2.0,
                             hop_size=1.0, n_mfcc=40):
    """
    Slide a window over *audio* and extract features from each frame.

    Returns
    -------
    feature_list : list of np.ndarray
    timestamps   : list of float  (start time of each window in seconds)
    """
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)

    feature_list, timestamps = [], []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        segment = audio[start: start + window_samples]
        features = _compute_features(segment, sample_rate, n_mfcc)
        feature_list.append(features)
        timestamps.append(start / sample_rate)

    return feature_list, timestamps


# ---------------------------------------------------------------------------
# CNN-focused: 2D mel spectrogram extraction
# ---------------------------------------------------------------------------

N_MELS_CNN = 128
HOP_LENGTH_CNN = 512


def extract_mel_spectrogram_2d(file_path, sample_rate=22050, duration=4.0,
                                n_mels=N_MELS_CNN, hop_length=HOP_LENGTH_CNN,
                                min_rms=0.001):
    """Load an audio file and return a (n_mels, time_frames) mel spectrogram
    normalized to [0, 1].  Returns None on error or near-silent audio."""
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)
        expected = int(sample_rate * duration)
        if len(audio) < expected:
            audio = np.pad(audio, (0, expected - len(audio)))
        else:
            audio = audio[:expected]
        if np.sqrt(np.mean(audio ** 2)) < min_rms:
            return None
        return _compute_mel_2d(audio, sr, n_mels, hop_length)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_mel_segments_from_long_file(file_path, sample_rate=22050, segment_duration=4.0,
                                         n_mels=N_MELS_CNN, hop_length=HOP_LENGTH_CNN,
                                         min_rms=0.001):
    """Slice a long audio file into non-overlapping *segment_duration*-second chunks
    and return one (n_mels, time_frames) mel spectrogram per kept chunk.

    Returns
    -------
    mel_list     : list of np.ndarray — one (n_mels, time_frames) array per valid segment
    n_segments   : int               — total segments before silence filter
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], 0

    seg_samples = int(sr * segment_duration)
    total_segments = len(audio) // seg_samples
    mel_list = []

    if total_segments == 0 and len(audio) > 0:
        # File shorter than one segment — pad and keep
        segment = np.pad(audio, (0, seg_samples - len(audio)))
        if np.sqrt(np.mean(audio ** 2)) >= min_rms:
            mel_list.append(_compute_mel_2d(segment, sr, n_mels, hop_length))
        return mel_list, 1

    for i in range(total_segments):
        segment = audio[i * seg_samples: (i + 1) * seg_samples]
        if np.sqrt(np.mean(segment ** 2)) < min_rms:
            continue
        mel_list.append(_compute_mel_2d(segment, sr, n_mels, hop_length))

    return mel_list, total_segments


def extract_mel_from_segment(audio, sample_rate=22050, duration=4.0,
                              n_mels=N_MELS_CNN, hop_length=HOP_LENGTH_CNN):
    """Return a (n_mels, time_frames) mel spectrogram for an already-loaded array.
    Pads or truncates to *duration* seconds before extraction."""
    expected = int(sample_rate * duration)
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    else:
        audio = audio[:expected]
    return _compute_mel_2d(audio, sample_rate, n_mels, hop_length)


def sliding_window_mel(audio, sample_rate=22050, window_size=4.0, hop_size=2.0,
                        n_mels=N_MELS_CNN, hop_length=HOP_LENGTH_CNN):
    """Slide a window over *audio* and return (mel_list, timestamps).

    Returns
    -------
    mel_list   : list of (n_mels, time_frames) np.ndarray
    timestamps : list of float — window start times in seconds
    """
    window_samples = int(window_size * sample_rate)
    hop_samples    = int(hop_size    * sample_rate)
    mel_list, timestamps = [], []

    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        segment = audio[start: start + window_samples]
        mel_list.append(_compute_mel_2d(segment, sample_rate, n_mels, hop_length))
        timestamps.append(start / sample_rate)

    return mel_list, timestamps


def _compute_mel_2d(audio, sr, n_mels=N_MELS_CNN, hop_length=HOP_LENGTH_CNN):
    """Compute a per-sample min-max normalized mel spectrogram (dB scale)."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min < 1e-6:
        return np.zeros_like(mel_db, dtype=np.float32)
    return ((mel_db - mel_min) / (mel_max - mel_min)).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helper (original MFCC pipeline)
# ---------------------------------------------------------------------------

def _compute_features(audio, sr, n_mfcc=40):
    features = []

    # MFCCs — 40 mean + 40 std = 80
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # Chroma — 12 mean + 12 std = 24
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Mel spectrogram — 128 mean + 128 std = 256
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # Spectral centroid — 2
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(float(np.mean(sc)))
    features.append(float(np.std(sc)))

    # Spectral rolloff — 2
    sr_feat = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(float(np.mean(sr_feat)))
    features.append(float(np.std(sr_feat)))

    # Zero crossing rate — 2
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    return np.array(features, dtype=np.float32)
