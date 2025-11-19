# Conversational AI - Commands Only

## Train Voice Model
```bash
python scripts/train_voice_model.py --data-file data/pitt_voice_metadata.csv --model-output models/voice_model.pkl --model-type random_forest
```

## Train Text Model
```bash
python scripts/train_text_model.py --data-file data/pitt_text_features.csv --model-output models/text_model.pkl --model-type random_forest
```

## Prepare Pitt dataset (text CSV)
```bash
python scripts/prepare_pitt_dataset.py --out data/pitt_text_features.csv
```

## Prepare audio metadata CSV (if you have audio files)
```bash
python -c "from src.data_loaders.voice_loader import save_audio_csv; save_audio_csv('data/pitt_voice_metadata.csv')"
```

## Cross-validation training (k-fold)
Run k-fold CV (example 5-fold) and retrain on full data:
```bash
python scripts/train_text_model.py --data-file data/pitt_text_features.csv --model-output models/text_model_cv.pkl --kfold 5
python scripts/train_voice_model.py --data-file data/pitt_voice_metadata.csv --model-output models/voice_model_cv.pkl --kfold 5
```

## Run API Server
```bash
python run_api.py
# then open http://localhost:8000/docs
```

## Voice Inference from Audio
```bash
python scripts/inference_voice.py --model models/voice_model.pkl --audio data/sample_audio.wav
```

## Voice Inference from JSON
```bash
python scripts/inference_voice.py --model models/voice_model.pkl --features data/voice_features.json
```

## Text Inference from Text
```bash
python scripts/inference_text.py --model models/text_model.pkl --text "transcript text here"
```

## Text Inference from File
```bash
python scripts/inference_text.py --model models/text_model.pkl --text-file data/transcript.txt
```

## Text Inference from JSON
```bash
python scripts/inference_text.py --model models/text_model.pkl --features data/text_features.json
```
