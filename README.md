# Dementia Detection System - Backend

A comprehensive machine learning system for detecting dementia risk through analysis of conversational speech patterns. The system analyzes both text transcripts and voice recordings to extract 10 key clinical parameters.

## 🎯 Overview

This system uses conversational AI to detect dementia risk by analyzing:
- **Text-based features** (7 parameters): Semantic incoherence, repeated questions, self-correction, low-confidence answers, hesitation pauses, emotion/slips, evening errors
- **Voice-based features** (3 parameters): Vocal tremors, slowed speech, in-session decline

### The 10 Detection Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| 1. Semantic Incoherence | Text | Illogical or off-topic utterances |
| 2. Repeated Questions | Text | Asking same question multiple times |
| 3. Self-Correction | Text | Instances of correcting oneself |
| 4. Low-Confidence Answers | Text | Hesitant or unsure responses |
| 5. Hesitation Pauses | Text | Filled pauses (um, uh, er, etc.) |
| 6. Vocal Tremors | Voice | Amplitude modulation at ~5 Hz |
| 7. Emotion + Slip | Text | Inappropriate emotional expressions |
| 8. Slowed Speech | Voice | Reduced speech rate |
| 9. Evening Errors | Text/Metadata | Time-dependent cognitive decline |
| 10. In-Session Decline | Voice | Progressive fatigue during session |

## 📁 Project Structure

```
dementia_backend/
├── config.py                          # Configuration management
├── requirements.txt                   # Python dependencies
├── run_api.py                        # FastAPI server launcher
├── test_prediction.py                # Test script with sample data
│
├── data/
│   ├── sample/                       # Sample dataset for testing
│   │   ├── audio/                   # Sample audio files (WAV, MP3, etc.)
│   │   ├── text/                    # Sample transcripts
│   │   └── metadata/
│   │       └── sample_data.json      # Sample metadata
│   ├── real/                        # Real dataset location (to be added)
│   └── generate_sample_audio.py     # Script to generate sample audio
│
├── src/
│   ├── api/
│   │   └── app.py                  # FastAPI application
│   │
│   ├── features/
│   │   ├── feature_extractor.py    # Generic feature extractor
│   │   └── conversational_ai/       # Chatbot-integrated features
│   │       ├── feature_extractor.py # Main extractor
│   │       └── components/
│   │           ├── text_processor.py    # Text analysis
│   │           └── voice_analyzer.py    # Audio analysis
│   │
│   ├── models/
│   │   ├── dementia_predictor.py   # Prediction model
│   │   └── conversational_ai/
│   │       └── model_utils.py      # Model utilities
│   │
│   ├── preprocessing/
│   │   └── data_loader.py          # Data loading and management
│   │
│   └── parsers/
│       └── __init__.py             # Data parsers
│
└── models/                          # Saved ML models (generated)
    └── dementia_predictor.pkl      # Trained model
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test with Sample Data

```bash
python test_prediction.py
```

This will:
- Load 5 sample cases (3 control, 2 dementia risk)
- Extract features from transcripts
- Display analysis reports
- Show how the system works

### 3. Start the API Server

```bash
python run_api.py
```

Access the API:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📊 API Endpoints

### Health & Info
- `GET /health` - Health check
- `GET /models` - Available models and feature info
- `GET /` - API overview

### Feature Extraction
- `POST /extract-features` - Extract all 10 features from text/audio

### Predictions
- `POST /predict` - Predict dementia risk from transcript + audio
- `POST /predict/text` - Predict from text only
- `POST /predict/audio` - Predict from audio file upload
- `POST /predict/batch` - Batch prediction for multiple patients

## 💾 Using the System

### With Sample Data (Current)

The system comes with 5 sample cases in `/data/sample/`:
- `sample_001.txt/wav` - Control (45F)
- `sample_002.txt/wav` - Dementia Risk (72M)
- `sample_003.txt/wav` - Control (68F)
- `sample_004.txt/wav` - Dementia Risk (76M)
- `sample_005.txt/wav` - Control (55M)

### Switching to Real Data (Future)

When you have real clinical data:

1. **Organize your data**:
   ```
   data/real/
   ├── audio/
   │   ├── patient_001.wav
   │   ├── patient_002.wav
   │   └── ...
   ├── text/
   │   ├── patient_001.txt
   │   ├── patient_002.txt
   │   └── ...
   └── metadata/
       └── dataset.json
   ```

2. **Create metadata file** (`data/real/metadata/dataset.json`):
   ```json
   {
     "version": "1.0",
     "samples": [
       {
         "id": "patient_001",
         "label": "control",
         "age": 65,
         "gender": "M",
         "audio_file": "audio/patient_001.wav",
         "transcript_file": "text/patient_001.txt"
       }
     ]
   }
   ```

3. **Update code to use real data**:
   ```python
   dataset_manager = DatasetManager()
   dataset_manager.switch_to_real_data()
   # Everything else works the same!
   ```

**No code changes needed!** The system is designed to seamlessly transition from sample to real data.

## 🔧 Feature Extraction Details

### Text Features (TextProcessor)

```python
from src.features.conversational_ai.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# Extract from transcript
features = extractor.extract_features_normalized(
    transcript_text="Your interview transcript here...",
    audio_path="path/to/audio.wav"
)

# Features dict contains all 10 parameters
print(features)  # {'semantic_incoherence': 0.15, ...}
```

### Voice Features (VoiceAnalyzer)

The system analyzes:
- **Vocal Tremors**: Detects amplitude modulation (~5 Hz)
- **Slowed Speech**: Analyzes spectral flux and speech rate
- **In-Session Decline**: Compares audio energy first half vs second half

## 📈 Making Predictions

### Text-Only Prediction

```bash
curl -X POST "http://localhost:8000/predict/text" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Your patient interview transcript...",
    "patient_id": "P001",
    "age": 72,
    "gender": "M"
  }'
```

### Audio File Upload

```bash
curl -X POST "http://localhost:8000/predict/audio" \
  -F "file=@interview.wav" \
  -F "transcript=Your transcript text..." \
  -F "patient_id=P001"
```

### Full Prediction (Text + Audio)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Your interview transcript...",
    "audio_file_path": "/path/to/audio.wav",
    "patient_id": "P001",
    "age": 72
  }'
```

## 🎓 Understanding the Output

### Feature Extraction Output

```json
{
  "semantic_incoherence": 0.15,      // 0-1, higher = more incoherent
  "repeated_questions": 0,            // count
  "self_correction": 2,               // count
  "low_confidence_answers": 0.1,      // 0-1, higher = less confident
  "hesitation_pauses": 3,             // count
  "vocal_tremors": 0.05,              // 0-1, higher = more tremor
  "emotion_slip": 0.2,                // 0-1, higher = more inappropriate
  "slowed_speech": 0.08,              // 0-1, higher = slower
  "evening_errors": 0.0,              // 0-1 (requires metadata)
  "in_session_decline": 0.0           // 0-1, higher = more decline
}
```

### Prediction Output

```json
{
  "patient_id": "P001",
  "prediction": "dementia_risk",              // or "control"
  "risk_score": 0.68,                         // 0-1 probability
  "confidence": 0.36,                         // how far from 0.5 threshold
  "features": {...},                          // all 10 features
  "feature_contributions": {...},             // weighted feature importance
  "recommendations": [                        // clinical recommendations
    "⚠️ High dementia risk detected...",
    "- Semantic incoherence detected...",
    "- Repetitive questioning observed..."
  ]
}
```

## 📝 Configuration

Edit `config.py` to customize:
- Audio sample rate (default: 16000 Hz)
- Feature thresholds
- API settings
- Model thresholds
- Logging settings

Environment variables:
```bash
export DATA_DIR=./data
export OUTPUT_DIR=./output
export MODELS_DIR=./models
export API_PORT=8000
export LOG_LEVEL=INFO
```

## 🔬 Testing & Validation

### Run Test Script
```bash
python test_prediction.py
```

### Generate Sample Audio
```bash
python data/generate_sample_audio.py
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Get feature info
curl http://localhost:8000/extract-features
```

## 📊 Dataset Statistics (Sample)

| Metric | Count |
|--------|-------|
| Total Cases | 5 |
| Control | 3 |
| Dementia Risk | 2 |
| Age Range | 45-76 years |
| Mean Age | 63.2 years |
| Males | 2 |
| Females | 3 |

## ⚙️ Model Architecture

The system uses an ensemble approach:
1. **Rule-based** (with sample data): Uses weighted feature scores
2. **ML-based** (future with real data):
   - Random Forest (100 trees)
   - Gradient Boosting (100 iterations)
   - Logistic Regression

Ensemble voting: Average of all models

## 🔄 Data Flow

```
Interview (Audio + Text)
       ↓
[Feature Extraction]
  - Text Processor: extracts 7 text features
  - Voice Analyzer: extracts 3 voice features
       ↓
[10 Feature Vector]
       ↓
[Dementia Predictor]
  - Rule-based (sample data)
  - ML-based (real data)
       ↓
[Risk Score + Classification]
  - dementia_risk (score > 0.5)
  - control (score ≤ 0.5)
       ↓
[Clinical Recommendations]
```

## 📚 Adding Real Data

When you have real clinical data:

1. Place audio files in `data/real/audio/`
2. Place transcripts in `data/real/text/`
3. Create metadata JSON file
4. Update dataset manager to use real data
5. All feature extraction and prediction code works unchanged

**Example workflow**:
```python
# Load real data
loader = DatasetManager()
loader.load_real_dataset_metadata("data/real/metadata/dataset.json")
loader.switch_to_real_data()

# Get samples
samples = loader.get_all_samples()

# Extract features - same code as sample data!
for sample in samples:
    features = extractor.extract_features_normalized(
        transcript_path=sample['transcript_path'],
        audio_path=sample['audio_path']
    )
    print(f"Patient {sample['id']}: {features}")
```

## 🛠️ Troubleshooting

### Missing Audio Libraries
```bash
pip install librosa soundfile numpy
```

### API Port Already in Use
```bash
python run_api.py --port 8001
```

### Feature Extraction Errors
- Ensure audio files are readable
- Check transcript encoding (UTF-8)
- Verify file paths are correct

## 📖 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI
- [Feature Details](src/features/conversational_ai/) - Component implementations
- [Model Architecture](src/models/) - Prediction models

## 👥 Contributing

To add new features:

1. Add extraction method to `TextProcessor` or `VoiceAnalyzer`
2. Update `FeatureExtractor.extract_features()`
3. Add to feature names list
4. Update documentation

## 📄 License

Research use only. Do not use for clinical diagnosis without proper validation.

## 📧 Contact

For questions or issues, contact the research team.

---

**Last Updated**: November 2025
**Status**: Sample data ready, waiting for real clinical data