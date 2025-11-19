#!/usr/bin/env python3
"""
Prepare Pitt Corpus Dataset

Parses .cha files under data/Pitt and extracts participant transcripts,
computes features using NLPEngine (preferred) or TextProcessor (fallback),
and writes a CSV with features and labels (dementia_label: 0=control,1=dementia).

Usage:
  python scripts/prepare_pitt_dataset.py --out data/pitt_text_features.csv
"""

import argparse
import csv
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.conversational_ai.nlp.nlp_engine import NLPEngine
from src.features.conversational_ai.components.text.text_processor import TextProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_cha_file(file_path: Path) -> str:
    """Extract participant utterances (*PAR:) from a CHAT .cha file."""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('*PAR:'):
                    # Remove speaker tag and timestamps if any
                    content = line.split('\t')[-1]
                    # remove inline annotations like  ൵123_456൵ (non-ascii markers)
                    content = content.replace('\u2028', ' ')
                    texts.append(content)
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
    return ' '.join(texts)


def collect_files(pitt_dir: Path):
    files = []
    for group in ['Control', 'Dementia']:
        group_dir = pitt_dir / group
        if not group_dir.exists():
            continue
        for task_dir in group_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for cha in task_dir.glob('*.cha'):
                files.append((group, task_dir.name, cha))
    return files


def main(out_file: str):
    pitt_dir = Path('data/Pitt')
    if not pitt_dir.exists():
        logger.error('data/Pitt not found in repository root')
        return

    # Initialize engines
    nlp = None
    try:
        nlp = NLPEngine()
        logger.info('NLPEngine initialized')
    except Exception:
        nlp = None

    text_proc = TextProcessor()

    files = collect_files(pitt_dir)
    logger.info(f'Found {len(files)} .cha files to process')

    rows = []
    for group, task, cha_path in files:
        participant_text = parse_cha_file(cha_path)
        if not participant_text.strip():
            continue

        # Extract features
        features = None
        if nlp:
            try:
                features = nlp.extract_dementia_markers(participant_text)
            except Exception as e:
                logger.warning(f'NLPEngine failed for {cha_path.name}: {e}')
                features = text_proc.process(participant_text)
        else:
            features = text_proc.process(participant_text)

        # Add metadata
        features['participant_id'] = cha_path.stem
        features['task'] = task
        features['file_path'] = str(cha_path)
        features['dementia_label'] = 1 if group.lower() == 'dementia' else 0

        rows.append(features)

    if not rows:
        logger.error('No data extracted, aborting')
        return

    # Determine columns (union of keys)
    columns = set()
    for r in rows:
        columns.update(r.keys())
    columns = list(columns)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    logger.info(f'Wrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Pitt corpus dataset')
    parser.add_argument('--out', default='data/pitt_text_features.csv', help='Output CSV file')
    args = parser.parse_args()
    main(args.out)
