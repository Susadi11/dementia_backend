"""
Start Dementia Detection API Server

Simple script to start the FastAPI server for dementia detection predictions
using conversational AI analysis.

Usage:
    python run_api.py

Then access:
    - API docs: http://localhost:8080/docs
    - Health: http://localhost:8080/health
    - Predictions: http://localhost:8080/predict
"""

import uvicorn
from pathlib import Path
import sys
import io
import os

# Suppress HuggingFace tokenizers fork warnings (safe to disable for inference-only server)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Main function to start the API server."""
    print("\n" + "="*80)
    print("Dementia Care Chatbot API Server")
    print("="*80)
    print("\nAPI Documentation:")
    print("  • Swagger UI: http://localhost:8080/docs")
    print("  • ReDoc:      http://localhost:8080/redoc")
    print("\nChatbot Endpoints:")
    print("  • Text Chat:       POST http://localhost:8080/chat/text")
    print("  • Voice Chat:      POST http://localhost:8080/chat/voice")
    print("  • Health Check:    GET  http://localhost:8080/chat/health")
    print("  • Session History: GET  http://localhost:8080/chat/sessions/{id}")
    print("  • Clear Session:   DEL  http://localhost:8080/chat/sessions/{id}")
    print("\nOther Endpoints:")
    print("  • Health Check:  http://localhost:8080/health")
    print("  • Predictions:   http://localhost:8080/predict")
    print("\nModel Information:")
    print("  • Base Model: LLaMA 3.2 1B Instruct")
    print("  • Training:   DailyDialog dataset")
    print("  • Method:     LoRA fine-tuning")
    print("  • Purpose:    Empathetic elderly care conversations")
    print("\n" + "="*80)
    print("Tip: Press CTRL+C to stop the server")
    print("="*80 + "\n")

    try:
        uvicorn.run(
            "src.api.app:app",
            host="0.0.0.0",
            port=8080,
            log_level="info",
            timeout_keep_alive=30
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        print("\nMake sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Created src/api/app.py with FastAPI application")
        print("  3. Set up your .env file with necessary configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()
