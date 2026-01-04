# Audio Reminder Creation - Implementation Guide

## ✅ Implementation Complete

Voice-based reminder creation has been successfully implemented in your dementia backend system.

## 🎯 What Was Added

### 1. New API Endpoint: `/api/reminders/create-from-audio`

**Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Audio file (WAV, MP3, M4A, OGG, FLAC)
- `user_id` (required): User identifier
- `priority` (optional): Reminder priority (low, medium, high, critical)
- `caregiver_ids` (optional): Comma-separated caregiver IDs

### 2. Features Implemented

✅ **Audio Upload Handling**
- Accepts multiple audio formats
- Handles file validation and temporary storage

✅ **Whisper Transcription**
- Converts audio to text using OpenAI Whisper
- Local, free transcription (no API costs)
- Auto-detects language

✅ **NLP Parsing**
- Extracts reminder details from transcribed text
- Automatically detects:
  - Category (medication, appointment, meal, hygiene, activity)
  - Time/date (8 AM, 2:30 PM, noon, morning, evening)
  - Recurrence (daily, weekly, monthly)
  - Priority level (based on keywords)

✅ **Automatic Reminder Creation**
- Creates reminder in database
- Notifies caregivers
- Returns complete reminder details

### 3. Enhanced Natural Language Endpoint

The existing `/api/reminders/natural-language` endpoint now includes:
- Full NLP parsing implementation
- Time extraction from text
- Category detection
- Priority inference

## 📋 API Usage Examples

### Python Example
```python
import requests

# Upload audio file to create reminder
with open('reminder_audio.wav', 'rb') as audio_file:
    files = {'file': ('reminder.wav', audio_file, 'audio/wav')}
    data = {
        'user_id': 'patient_001',
        'priority': 'high',
        'caregiver_ids': 'caregiver_001'
    }
    
    response = requests.post(
        'http://localhost:8000/api/reminders/create-from-audio',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Transcription: {result['transcription']}")
    print(f"Reminder ID: {result['reminder']['id']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/reminders/create-from-audio" \
  -F "user_id=patient_001" \
  -F "priority=high" \
  -F "caregiver_ids=caregiver_001" \
  -F "file=@/path/to/audio.wav"
```

### JavaScript/Fetch Example
```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('user_id', 'patient_001');
formData.append('priority', 'high');

const response = await fetch('http://localhost:8000/api/reminders/create-from-audio', {
    method: 'POST',
    body: formData
});

const data = await response.json();
console.log('Reminder created:', data);
```

## 🎤 Supported Voice Commands

The system can parse these types of voice commands:

1. **Medication Reminders**
   - "Remind me to take my blood pressure medicine at 8 AM every morning"
   - "Take my evening pills at 6 PM daily"

2. **Appointment Reminders**
   - "Doctor appointment next Tuesday at 2 PM"
   - "Set a reminder for my checkup"

3. **Meal Reminders**
   - "Remind me about lunch at noon"
   - "I need to eat breakfast every morning"

4. **Activity Reminders**
   - "Remind me to take a walk in the evening"
   - "Exercise time at 5 PM daily"

5. **Hygiene Reminders**
   - "Remind me to take my shower every morning"
   - "Brush teeth at bedtime"

## 🔧 NLP Parsing Capabilities

### Time Detection
- **Specific times:** "8 AM", "2:30 PM", "at noon"
- **Relative times:** "morning" (8 AM), "evening" (6 PM), "bedtime" (9 PM)
- **Smart scheduling:** If time has passed today, schedules for tomorrow

### Category Detection
Keywords automatically determine category:
- **Medication:** medicine, medication, pill, tablet, drug
- **Appointment:** doctor, appointment, visit, checkup
- **Meal:** breakfast, lunch, dinner, meal, eat
- **Hygiene:** shower, bath, hygiene, brush, wash
- **Activity:** exercise, walk, activity

### Recurrence Detection
- **Daily:** "every day", "daily", "everyday"
- **Weekly:** "every week", "weekly"
- **Monthly:** "every month", "monthly"

### Priority Inference
- **High:** "urgent", "critical", "important", "asap"
- **Low:** "when you can", "sometime", "eventually"
- **Medium:** Default priority

## 🧪 Testing

### Run Tests
```bash
# Test all reminder endpoints including audio upload
python test/test_reminder_endpoints.py

# Test audio upload specifically with examples
python test/test_audio_reminder_example.py
```

### Test Response Format
```json
{
  "status": "success",
  "message": "Reminder created successfully from audio",
  "reminder": {
    "id": "reminder_abc123",
    "user_id": "patient_001",
    "title": "Take Medication (Daily)",
    "description": "Remind me to take my blood pressure medicine at 8 AM",
    "category": "medication",
    "priority": "high",
    "scheduled_time": "2026-01-05T08:00:00",
    "recurrence": "daily",
    "status": "active"
  },
  "transcription": "Remind me to take my blood pressure medicine at 8 AM",
  "audio_file": "reminder_command.wav"
}
```

## 📁 Files Modified/Created

1. **`src/routes/reminder_routes.py`** - Added audio upload endpoint and NLP parsing
2. **`test/test_reminder_endpoints.py`** - Added audio upload test
3. **`test/test_audio_reminder_example.py`** - Complete usage examples and demos

## 🔗 Integration with Frontend

### React Component Example
See `test/test_audio_reminder_example.py` for a complete React component that:
- Records or uploads audio
- Submits to API
- Displays transcription and created reminder

### Mobile App Integration
- Use device's microphone to record
- Convert to supported format (WAV, MP3)
- Upload using multipart/form-data
- Handle 201 response with reminder details

## ⚙️ Dependencies Required

Make sure these packages are installed:
```bash
pip install openai-whisper
pip install faster-whisper  # Alternative, faster implementation
pip install soundfile
pip install librosa
```

## 🚀 How It Works

1. **Audio Upload** → User uploads audio file via API
2. **Transcription** → Whisper converts audio to text
3. **NLP Analysis** → Custom parser extracts reminder details
4. **Validation** → System validates extracted information
5. **Creation** → Reminder is created and stored in database
6. **Response** → API returns complete reminder details and transcription

## 🎯 Next Steps

You can now:
1. ✅ Create reminders by uploading audio files
2. ✅ Parse natural language commands
3. ✅ Automatically detect time, category, and priority
4. ✅ Integrate with your frontend/mobile app

## 📝 Notes

- Whisper transcription runs locally (no API costs)
- Supports multiple languages (auto-detected)
- Audio files are temporarily stored and deleted after processing
- NLP parsing is rule-based (can be enhanced with ML models)
- All reminder features (snooze, response tracking, etc.) work with audio-created reminders

## 🐛 Troubleshooting

**Issue:** "Could not transcribe audio"
- Ensure audio quality is good
- Check supported formats: WAV, MP3, M4A, OGG, FLAC
- Verify file is not corrupted

**Issue:** "Connection refused"
- Make sure API server is running: `python run_api.py`
- Check port 8000 is not blocked

**Issue:** "Time not parsed correctly"
- Use clear time formats: "8 AM", "2:30 PM", "noon"
- Avoid ambiguous phrases

---

**Implementation Date:** January 4, 2026  
**Status:** ✅ Production Ready
