## 📁 Project Structure

```
dementia_backend/
│
├── 📄 Root Files
│   ├── config.py                     # App configuration settings
│   ├── requirements.txt              # Python package dependencies
│   ├── run_api.py                    # Start the API server
│   └── test_prediction.py            # Test script
│
├── 📂 data/
│   ├── sample/                       # Sample test data
│   │   ├── audio/                    # Sample audio files
│   │   ├── text/                     # Sample transcript files
│   │   └── metadata/                 # Sample metadata
│   └── generate_sample_audio.py      # Audio generation script
│
├── 📂 logs/                          # Application log files
│
├── 📂 models/                        # Saved trained models
│
├── 📂 output/                        # Generated output results
│
└── 📂 src/                           # Main application code
    │
    ├── 📂 api/
    │   ├── __init__.py
    │   └── app.py                    # FastAPI main application
    │
    ├── 📂 routers/                   # API endpoint definitions
    │   ├── __init__.py
    │   ├── healthcheck.py            # Health check endpoints
    │   └── conversational_ai.py       # Chat endpoints (text/voice)
    │
    ├── 📂 services/                  # Business logic layer
    │   ├── __init__.py
    │   ├── db_service.py             # Database operations
    │   ├── user_service.py           # User management
    │   └── session_service.py        # Session management (coming soon)
    │
    ├── 📂 features/                  # Feature extraction logic
    │   ├── __init__.py
    │   ├── base_features.py
    │   ├── conversational_ai/        # Chatbot features
    │   │   ├── __init__.py
    │   │   ├── conversational.py
    │   │   ├── feature_extractor.py
    │   │   └── components/
    │   │       ├── __init__.py
    │   │       ├── text_processor.py
    │   │       └── voice_analyzer.py
    │   ├── mmse/                     # MMSE test features (coming soon)
    │   ├── games/                    # Game features (coming soon)
    │   ├── reminders/                # Reminder features (coming soon)
    │   └── shared_utils/             # Shared utility functions
    │
    ├── 📂 models/                    # ML models and training
    │   ├── __init__.py
    │   └── conversational_ai/        # Conversational AI models
    │       ├── __init__.py
    │       ├── model_trainer.py
    │       ├── model_utils.py
    │       └── trained_models/       # Saved model files
    │
    ├── 📂 preprocessing/             # Data preprocessing
    │   ├── __init__.py
    │   ├── preprocessor.py
    │   ├── data_loader.py
    │   ├── data_cleaner.py
    │   ├── data_validator.py
    │   ├── feature_selector.py
    │   ├── audio_models.py
    │   └── voice_processor.py
    │
    ├── 📂 parsers/                   # Data parsing utilities
    │   ├── __init__.py
    │   └── chat_parser.py
    │
    └── 📂 utils/                     # Helper functions
        ├── __init__.py
        ├── logger.py                 # Logging setup
        └── helpers.py                # Utility functions
```

## 📚 Folder Descriptions

| Folder | Purpose |
|--------|---------|
| **routers/** | API endpoint routes (what users call) |
| **services/** | Business logic and database operations |
| **features/** | Extract features from text & voice |
| **models/** | ML models and predictions |
| **preprocessing/** | Clean & prepare data |
| **utils/** | Helper functions and logging |

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python run_api.py
```

The API will be available at: `http://localhost:8000`

### 3. View API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 Current API Endpoints

### Health Check
- `GET /health` - API status
- `GET /health/status` - Detailed system status

### Chat (Conversational AI)
- `POST /chat/text` - Send text message
- `POST /chat/voice` - Send voice message (audio file)
- `GET /chat/sessions/{session_id}` - Get conversation history
- `DELETE /chat/sessions/{session_id}` - Clear conversation


