"""
Simple test for audio reminder endpoint
"""
import requests
import io
import wave
import numpy as np

def test_audio_endpoint():
    # Create simple test audio
    sample_rate = 16000
    duration = 1
    audio_data = np.zeros(sample_rate * duration, dtype=np.int16)
    
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    audio_buffer.seek(0)
    
    # Test the endpoint
    files = {'file': ('test.wav', audio_buffer, 'audio/wav')}
    data = {'user_id': 'patient_001', 'priority': 'high'}
    
    try:
        response = requests.post(
            'http://localhost:8000/api/reminders/create-from-audio',
            files=files,
            data=data,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json() if response.status_code != 500 else response.text}")
        
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_audio_endpoint()
    print(f"\n{'✅ Success' if success else '❌ Failed'}")
