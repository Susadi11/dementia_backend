"""
Unified Voice Data Loader

Aggregates audio metadata into a single CSV. For the Pitt corpus there may be
no audio files in the repo; this loader will look for common audio directories
and produce a CSV with audio_path, participant_id, task, duration_seconds,
sample_rate, dementia_label.
"""
from pathlib import Path
from typing import List
import logging

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False


def inspect_audio_file(path: Path) -> dict:
    if not HAS_LIBROSA:
        return {'duration_seconds': None, 'sample_rate': None}
    try:
        y, sr = librosa.load(str(path), sr=None)
        duration = len(y) / sr if sr and len(y) else 0.0
        return {'duration_seconds': round(duration, 2), 'sample_rate': int(sr)}
    except Exception as e:
        logger.warning(f"Failed to inspect {path}: {e}")
        return {'duration_seconds': None, 'sample_rate': None}


def collect_audio_files(data_root: str = 'data') -> pd.DataFrame:
    root = Path(data_root)
    rows: List[dict] = []

    # Search for audio files recursively (common extensions)
    exts = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    for ext in exts:
        for p in root.rglob(ext):
            # Heuristic: participant id from filename (before first dash) or parent folder
            participant = p.stem.split('-')[0]
            task = p.parent.name
            meta = inspect_audio_file(p)
            rows.append({
                'audio_path': str(p),
                'participant_id': participant,
                'task': task,
                'duration_seconds': meta.get('duration_seconds'),
                'sample_rate': meta.get('sample_rate'),
                'dementia_label': None
            })

    df = pd.DataFrame(rows)
    return df


def save_audio_csv(out_path: str = 'data/pitt_voice_metadata.csv') -> None:
    df = collect_audio_files()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} audio rows to {out}")


if __name__ == '__main__':
    save_audio_csv()
