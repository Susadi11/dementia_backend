# Quick Start Guide - Reminder System

## 🚀 Get Started in 5 Minutes

### Step 1: Verify Installation

```bash
# Check Python version (3.8+ required)
python --version

# Verify dependencies
pip install -r requirements.txt
```

### Step 2: Test the System

```bash
# Run test suite to verify everything works
python test_reminder_system.py
```

Expected output:
```
================================================================================
  CONTEXT-AWARE SMART REMINDER SYSTEM - TEST SUITE
================================================================================

Testing Reminder Response Analyzer
...
✓ Reminder analyzer test completed

Testing Behavior Tracker
...
✓ Behavior tracker test completed

ALL TESTS COMPLETED SUCCESSFULLY ✓
```

### Step 3: Start the API

```bash
# Start FastAPI server
python src/api/app.py
```

Or with uvicorn:
```bash
uvicorn src.api.app:app --reload --port 8000
```

### Step 4: Test API Endpoints

Open browser: http://localhost:8000/docs

Try the following:

#### 4.1 Create a Reminder
```bash
curl -X POST "http://localhost:8000/api/reminders/create" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "title": "Take morning medication",
    "scheduled_time": "2025-11-26T08:00:00",
    "priority": "critical",
    "category": "medication"
  }'
```

#### 4.2 Process User Response
```bash
curl -X POST "http://localhost:8000/api/reminders/respond" \
  -H "Content-Type: application/json" \
  -d '{
    "reminder_id": "rem123",
    "user_id": "test_user",
    "response_text": "Yes, I took my medication"
  }'
```

#### 4.3 Get Behavior Analytics
```bash
curl "http://localhost:8000/api/reminders/behavior/test_user?days=7"
```

### Step 5: (Optional) Train on Pitt Corpus

```bash
# Extract features from Pitt Corpus
python scripts/prepare_pitt_dataset.py --out output/pitt_features.csv

# Train model
python scripts/train_text_model.py \
  --data-file output/pitt_features.csv \
  --model-output models/conversational_ai/reminder_model.pkl
```

## 📝 Example Code

### Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Create reminder
reminder_data = {
    "user_id": "patient123",
    "title": "Take evening medication",
    "scheduled_time": "2025-11-25T20:00:00",
    "priority": "critical",
    "category": "medication",
    "caregiver_ids": ["caregiver456"]
}

response = requests.post(f"{BASE_URL}/api/reminders/create", json=reminder_data)
print("Reminder created:", response.json())

# 2. Process user response
response_data = {
    "reminder_id": "rem123",
    "user_id": "patient123",
    "response_text": "I don't remember... did I already do it?"
}

response = requests.post(f"{BASE_URL}/api/reminders/respond", json=response_data)
result = response.json()

print(f"Cognitive Risk: {result['cognitive_risk_score']}")
print(f"Action: {result['recommended_action']}")
print(f"Caregiver Notified: {result['caregiver_notified']}")

# 3. Get behavior analytics
response = requests.get(f"{BASE_URL}/api/reminders/behavior/patient123?days=30")
pattern = response.json()['pattern']

print(f"Confirmation Rate: {pattern['confirmed_count']}/{pattern['total_reminders']}")
print(f"Confusion Trend: {pattern['confusion_trend']}")
```

### JavaScript Client Example

```javascript
const BASE_URL = 'http://localhost:8000';

// Create reminder
async function createReminder() {
  const response = await fetch(`${BASE_URL}/api/reminders/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: 'patient123',
      title: 'Take morning medication',
      scheduled_time: '2025-11-26T08:00:00',
      priority: 'critical',
      category: 'medication'
    })
  });
  
  const data = await response.json();
  console.log('Reminder created:', data);
}

// Process user response
async function processResponse(reminderId, responseText) {
  const response = await fetch(`${BASE_URL}/api/reminders/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reminder_id: reminderId,
      user_id: 'patient123',
      response_text: responseText
    })
  });
  
  const result = await response.json();
  console.log('Cognitive Risk:', result.cognitive_risk_score);
  console.log('Action:', result.recommended_action);
}

// Get behavior analytics
async function getAnalytics(userId) {
  const response = await fetch(`${BASE_URL}/api/reminders/behavior/${userId}?days=30`);
  const data = await response.json();
  console.log('Behavior Pattern:', data.pattern);
}
```

## 🔍 Testing Scenarios

### Scenario 1: Normal Confirmation
```python
response = "Yes, I took my medication"
# Expected: interaction_type = "confirmed", low cognitive risk
```

### Scenario 2: Confusion Detected
```python
response = "What medicine? I don't understand"
# Expected: confusion_detected = True, caregiver_alert = True
```

### Scenario 3: Memory Issue
```python
response = "I don't remember... did I already do it?"
# Expected: memory_issue_detected = True, action = "provide_context_and_repeat"
```

### Scenario 4: Deliberate Delay
```python
response = "Later, I'm busy right now"
# Expected: interaction_type = "delayed", reminder rescheduled
```

## 📊 Expected Outputs

### Analyzer Output
```json
{
  "cognitive_risk_score": 0.75,
  "confusion_detected": true,
  "memory_issue_detected": true,
  "uncertainty_detected": false,
  "interaction_type": "confused",
  "recommended_action": "escalate_to_caregiver",
  "caregiver_alert_needed": true,
  "features": {
    "semantic_incoherence": 0.6,
    "repeated_questions": 0.8,
    "hesitation_pauses": 0.4
  }
}
```

### Behavior Pattern Output
```json
{
  "total_reminders": 20,
  "confirmed_count": 12,
  "ignored_count": 3,
  "confused_count": 5,
  "avg_cognitive_risk_score": 0.45,
  "confusion_trend": "declining",
  "optimal_reminder_hour": 8,
  "recommended_frequency_multiplier": 1.3,
  "escalation_recommended": true
}
```

## 🛠️ Troubleshooting

### Issue: Import errors
**Solution:** Ensure you're running from project root
```bash
cd "d:\4 th year 1st\researchbackend\dementia_backend"
python test_reminder_system.py
```

### Issue: API won't start
**Solution:** Check if port 8000 is available
```bash
# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess
```

### Issue: Model not found
**Solution:** The system works without trained model, but for best results:
```bash
python scripts/prepare_pitt_dataset.py --out output/pitt_features.csv
python scripts/train_text_model.py --data-file output/pitt_features.csv
```

## 📚 Documentation

- Full Documentation: `REMINDER_SYSTEM.md`
- Implementation Details: `IMPLEMENTATION_SUMMARY.md`
- API Docs: http://localhost:8000/docs
- Code Examples: `test_reminder_system.py`

## ✅ Verification Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tests pass (`python test_reminder_system.py`)
- [ ] API starts (`python src/api/app.py`)
- [ ] Swagger UI accessible (http://localhost:8000/docs)
- [ ] Can create reminder via API
- [ ] Can process user response
- [ ] Behavior analytics working

## 🎉 Success!

You're ready to use the Context-Aware Smart Reminder System!

**Next Steps:**
1. Integrate with your mobile app
2. Connect to database
3. Train on Pitt Corpus for improved accuracy
4. Add push notification service
5. Customize for your specific use case

For questions or issues, refer to the full documentation in `REMINDER_SYSTEM.md`.
