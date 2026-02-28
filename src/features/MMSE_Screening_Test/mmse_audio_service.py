# src/features/mmse_screening/mmse_audio_service.py

import os
import subprocess
import uuid
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
from fastapi import UploadFile, HTTPException

from .mmse_ml_service import transcribe_audio


def _convert_to_wav(input_path: str, sr: int = 16000) -> str:
    """
    Convert any input audio format to mono 16kHz wav using ffmpeg.
    """
    out_path = f"{input_path}_{uuid.uuid4().hex}.wav"

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(sr), out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail="Failed to convert audio") from exc

    return out_path


def _extract_audio_features(path: str, sr_target: int = 16000) -> np.ndarray:
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    y = librosa.util.normalize(y)

    if len(y) < sr:
        raise HTTPException(status_code=400, detail="Audio too short – record at least 1 second")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=sr
    )

    features = np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_delta.mean(axis=1),
        mfcc_delta2.mean(axis=1),
        chroma.mean(axis=1),
        contrast.mean(axis=1),
        tonnetz.mean(axis=1),
    ])

    return features.reshape(1, -1)


async def process_audio_upload(file: UploadFile):
    """
    Full audio pipeline for one uploaded file:
    - Save to temp file
    - Convert to wav
    - Extract audio features
    - Transcribe (Whisper)

    Returns:
        (audio_features, raw_transcript)
    """
    input_path = None
    wav_path = None

    try:
        # 1. Persist upload to temp file
        suffix = os.path.splitext(file.filename or "")[1] or ".tmp"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name

        # 2. Convert and extract + transcribe
        wav_path = _convert_to_wav(input_path)
        audio_features = _extract_audio_features(wav_path)
        raw_transcript = transcribe_audio(wav_path)

        return audio_features, raw_transcript

    finally:
        # Cleanup
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)