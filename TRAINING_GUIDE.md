# Context-Aware Smart Reminder System - Training Guide

## 🎯 Overview

This guide shows you how to train your context-aware smart reminder system using **synthetic data combined with the Pitt Corpus dataset** for both **text and voice modalities**.

## 📊 Datasets You're Using

### 1. **Pitt Corpus Dataset** (Primary)
- **Location:** `data/Pitt/`  
- **Content:** Real dementia patient speech patterns from DementiaBank
- **Structure:** Control vs Dementia groups across 4 tasks (cookie, fluency, recall, sentence)
- **Features:** 11 cognitive indicators extracted from `.cha` files

### 2. **Synthetic Reminder Dataset** (Generated)
- **Purpose:** Realistic reminder-response scenarios
- **Content:** 5 reminder categories × 5 cognitive levels
- **Features:** Text patterns + voice characteristics + cognitive markers

### 3. **Combined Multimodal Dataset** (Training)
- **Combination:** Pitt Corpus patterns + Synthetic scenarios
- **Modalities:** Text features + Voice features + Context features
- **Output:** Comprehensive training dataset optimized for reminder contexts

## 🚀 Quick Start Training

### Option 1: **One-Command Full Pipeline** (Recommended)

```bash
# Run complete training pipeline (generates 2000 synthetic samples)
python scripts/master_training_pipeline.py --full-pipeline

# With more samples and audio generation
python scripts/master_training_pipeline.py --full-pipeline --num-samples 5000 --generate-audio
```

This will automatically:
1. ✅ Process Pitt Corpus data
2. ✅ Generate synthetic reminder scenarios  
3. ✅ Combine datasets
4. ✅ Create voice features
5. ✅ Train all models
6. ✅ Validate results

### Option 2: **Step-by-Step Training**

```bash
# Step 1: Prepare Pitt Corpus features
python scripts/prepare_pitt_dataset.py --out output/pitt_text_features.csv

# Step 2: Generate synthetic reminder data
python -c "
from scripts.generate_synthetic_reminder_data import ReminderSyntheticDataGenerator
generator = ReminderSyntheticDataGenerator()
generator.generate_dataset(2000, 'data/synthetic_reminder_data.csv')
"

# Step 3: Generate voice features
python scripts/generate_voice_reminder_data.py \
  --text-data data/synthetic_reminder_data.csv \
  --output-voice data/voice_features.csv \
  --output-combined data/multimodal_dataset.csv

# Step 4: Train models
python scripts/train_reminder_models.py --data data/multimodal_dataset.csv
```

## 🧠 Models Trained

Your system will train **4 specialized models**:

### 1. **Confusion Detection Model**
- **Purpose:** Detect confusion in user responses
- **Input:** Text features + temporal features
- **Output:** Binary classification (confused/clear)
- **Use:** Real-time response analysis

### 2. **Cognitive Risk Assessment Model**
- **Purpose:** Calculate cognitive impairment risk score
- **Input:** Text + voice + interaction features
- **Output:** Risk score (0.0 - 1.0)
- **Use:** Trend monitoring and early detection

### 3. **Caregiver Alert Model**
- **Purpose:** Predict when caregiver intervention needed
- **Input:** All features + context
- **Output:** Alert needed (yes/no) + severity
- **Use:** Automatic caregiver notifications

### 4. **Response Type Classifier**
- **Purpose:** Classify response patterns
- **Input:** Multimodal features
- **Output:** Response type (clear/mild/moderate/high confusion)
- **Use:** Adaptive scheduling and personalization

## 📈 Training Data Distribution

### **Synthetic Data Categories**
- 🏥 **Medication reminders** (40%) - Most critical
- 🍽️ **Meal reminders** (20%) - Daily routine
- 👨‍⚕️ **Appointment reminders** (20%) - Healthcare
- 🛁 **Hygiene reminders** (10%) - Personal care
- 🔒 **Safety reminders** (10%) - Safety checks

### **Cognitive Levels**
- 😊 **Clear confirmation** (30%) - Normal responses
- 😕 **Mild confusion** (25%) - Slight uncertainty
- 😟 **Moderate confusion** (25%) - Memory issues
- 😰 **High confusion** (15%) - Significant impairment
- ⏰ **Delay/resistance** (5%) - Procrastination

## 🔧 Advanced Configuration

### **Custom Data Distribution**
```python
from scripts.generate_synthetic_reminder_data import ReminderSyntheticDataGenerator

generator = ReminderSyntheticDataGenerator()

# Custom distribution
custom_distribution = {
    "categories": {
        "medication": 0.6,  # More medication focus
        "meal": 0.2,
        "appointment": 0.1,
        "hygiene": 0.05,
        "safety": 0.05
    },
    "cognitive_levels": {
        "clear_confirmation": 0.2,
        "mild_confusion": 0.3,   # More confusion samples
        "moderate_confusion": 0.3,
        "high_confusion": 0.2
    }
}

generator.generate_dataset(
    num_samples=3000,
    distribution=custom_distribution
)
```

### **Voice Feature Configuration**
```python
from scripts.generate_voice_reminder_data import VoiceReminderDataGenerator

voice_gen = VoiceReminderDataGenerator()

# Generate with custom voice characteristics
voice_gen.voice_characteristics["mild_confusion"]["speech_rate"] = (80, 120)  # Slower speech
voice_gen.voice_characteristics["mild_confusion"]["pause_frequency"] = (0.4, 0.7)  # More pauses
```

## 📊 Expected Results

After training, you should see:

```
TRAINING RESULTS SUMMARY
=====================================

CONFUSION_DETECTION:
  ✅ AUC Score: ~0.85-0.95
  ✅ Model: Random Forest or Gradient Boosting

COGNITIVE_RISK:
  ✅ R² Score: ~0.75-0.90
  ✅ Model: Random Forest Regressor

CAREGIVER_ALERT:
  ✅ AUC Score: ~0.80-0.92
  ✅ Model: Random Forest Classifier

RESPONSE_CLASSIFIER:
  ✅ Accuracy: ~0.70-0.85
  ✅ Model: Multi-class Random Forest
```

## 🎮 Testing Your Trained Models

### **1. Real-time Testing**
```bash
# Start API with trained models
python src/api/app_simple.py

# Test real-time system
python test_realtime_system.py
```

### **2. Web Interface Testing**
```bash
# Open in browser
test_realtime_web.html
```

### **3. Programmatic Testing**
```python
from scripts.train_reminder_models import ReminderSystemTrainer

# Load trained models
trainer = ReminderSystemTrainer("models/reminder_system")
trainer.load_trained_models()

# Test with sample response
response_features = {
    'hesitation_pauses': 5,
    'semantic_incoherence': 0.6,
    'response_time_seconds': 45.0,
    'low_confidence_answers': 0.8
}

# Get predictions
confusion_prediction = trainer.models['confusion_detection'].predict([list(response_features.values())])
risk_score = trainer.models['cognitive_risk'].predict([list(response_features.values())])
```

## 🔄 Continuous Training

### **Adding New Data**
```bash
# Add new synthetic scenarios
python -c "
generator = ReminderSyntheticDataGenerator()
new_data = generator.generate_dataset(500, 'data/new_scenarios.csv')
"

# Retrain models with expanded data
python scripts/train_reminder_models.py --data data/expanded_dataset.csv
```

### **Model Updates**
- Retrain monthly with new interaction data
- Adjust distributions based on real usage patterns
- Fine-tune thresholds based on caregiver feedback

## 🎯 Integration with Your System

The trained models integrate directly with your reminder system:

1. **Real-time Analysis:** `PittBasedReminderAnalyzer` uses trained models
2. **Adaptive Scheduling:** `AdaptiveReminderScheduler` uses cognitive risk scores  
3. **Caregiver Alerts:** `CaregiverNotifier` uses alert prediction models
4. **Behavior Tracking:** `BehaviorTracker` uses response classifications

## 📁 Output Files

After training, you'll have:

```
data/training_pipeline/
├── synthetic_reminder_data.csv     # Generated scenarios
├── combined_text_training_data.csv # Text + Pitt data
├── voice_reminder_features.csv     # Voice features
├── multimodal_training_data.csv    # Final training set
└── pipeline_log.json              # Training log

models/reminder_system/
├── confusion_detection_model.joblib
├── cognitive_risk_model.joblib
├── caregiver_alert_model.joblib
├── response_classifier_model.joblib
└── training_metadata.json
```

## 🚨 Troubleshooting

### **Common Issues:**

1. **Missing Pitt Data:**
   ```bash
   # Ensure Pitt corpus is in data/Pitt/
   ls data/Pitt/Control/cookie/  # Should show .cha files
   ```

2. **Training Failures:**
   ```bash
   # Check pipeline log
   cat data/training_pipeline/pipeline_log.json
   ```

3. **Model Performance Issues:**
   - Increase `num_samples` (try 5000+)
   - Adjust cognitive level distributions
   - Check feature correlation

Your context-aware smart reminder system is now ready for comprehensive training with both Pitt Corpus real data and synthetic reminder scenarios! 🧠✨