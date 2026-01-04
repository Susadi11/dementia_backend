"""
Quick Start: Audio Reminder Creation
=====================================

Test the new audio upload feature immediately!
"""

import requests

def quick_test():
    """Quick test with minimal code."""
    import io
    import wave
    import numpy as np
    
    print("🎤 Quick Audio Reminder Test\n")
    
    # Create test audio (1 second silence)
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    audio_buffer.seek(0)
    
    # Upload to API
    try:
        response = requests.post(
            "http://localhost:8000/api/reminders/create-from-audio",
            files={'file': ('test.wav', audio_buffer, 'audio/wav')},
            data={'user_id': 'patient_001', 'priority': 'high'},
            timeout=30
        )
        
        if response.status_code == 201:
            result = response.json()
            print("✅ SUCCESS!\n")
            print(f"Transcription: {result['transcription']}")
            print(f"Reminder Title: {result['reminder']['title']}")
            print(f"Category: {result['reminder']['category']}")
            print(f"Scheduled: {result['reminder']['scheduled_time']}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ API not running. Start it with: python run_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
