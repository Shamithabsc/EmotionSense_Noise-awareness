from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import DATA_DIR, EMOTION_CODES, SPEECH_MODALITY

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def parse_ravdess_filename(path: Path) -> dict | None:
    """Parse the standard RAVDESS filename, e.g. 03-01-05-01-02-01-12.wav.

    Returns None for:
    - Filenames that do not match the 7-part RAVDESS convention.
    - Non-speech modalities (songs, video-only).  Only modality 03
      (audio-only speech) is accepted.
    - Unknown emotion codes.
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    # Enforce speech-only modality.
    if parts[0] != SPEECH_MODALITY:
        return None
    if parts[2] not in EMOTION_CODES:
        return None
    try:
        actor_id = int(parts[6])
    except ValueError:
        return None
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": parts[2],
        "emotion": EMOTION_CODES[parts[2]],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor_id": actor_id,
    }


def discover_ravdess(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Discover RAVDESS speech-only files recursively and return parsed metadata.

    Only files matching the 7-part filename convention AND having modality 03
    (audio-only speech) are included.  Song and video files are silently skipped.
    """
    data_dir = Path(data_dir)
    rows = []
    if not data_dir.exists():
        return pd.DataFrame()
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            row = parse_ravdess_filename(path)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def dataset_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(columns=["emotion", "files"]), pd.DataFrame(columns=["actor_id", "files"])
    by_emotion = df.groupby("emotion").size().rename("files").reset_index().sort_values("emotion")
    by_actor = df.groupby("actor_id").size().rename("files").reset_index().sort_values("actor_id")
    return by_emotion, by_actor
