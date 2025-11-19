"""
Unified Text Data Loader

Parses .cha files and sample text locations and returns a single pandas DataFrame
with columns: participant_id, task, file_path, text, dementia_label

This module is purposely small and robust to missing fields.
"""
from pathlib import Path
from typing import List
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def parse_cha_file(file_path: Path) -> str:
    texts: List[str] = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('*PAR:'):
                    # Take content after speaker tag
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        content = parts[-1]
                    else:
                        content = line.replace('*PAR:', '').strip()
                    # basic cleanup
                    content = content.replace('\u2028', ' ')
                    texts.append(content)
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
    return ' '.join(texts)


def collect_pitt_texts(pitt_root: str = 'data/Pitt') -> pd.DataFrame:
    root = Path(pitt_root)
    rows = []
    if not root.exists():
        logger.error(f"Pitt root not found: {root}")
        return pd.DataFrame()

    for group in ['Control', 'Dementia']:
        group_dir = root / group
        if not group_dir.exists():
            continue
        for task_dir in group_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for cha in task_dir.glob('*.cha'):
                text = parse_cha_file(cha)
                if not text.strip():
                    continue
                rows.append({
                    'participant_id': cha.stem,
                    'task': task_dir.name,
                    'file_path': str(cha),
                    'text': text,
                    'dementia_label': 1 if group.lower() == 'dementia' else 0
                })

    df = pd.DataFrame(rows)
    return df


def save_pitt_texts_csv(out_path: str = 'data/pitt_text_all.csv') -> None:
    df = collect_pitt_texts()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} rows to {out}")


if __name__ == '__main__':
    save_pitt_texts_csv()
