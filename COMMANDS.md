# Conversational AI - Commands Only

## Train Voice Model
```bash
python scripts/train_voice_model.py --data-file data/voice_features.csv --model-output models/voice_model.pkl --model-type random_forest
```

## Train Text Model
```bash
python scripts/train_text_model.py --data-file data/text_features.csv --model-output models/text_model.pkl --model-type random_forest
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
