
# Pitt Corpus Integration Guide

This guide shows you how to integrate the Pitt Corpus dementia data with your synthetic reminder data to create more robust, real-world trained models.

## 🎯 Overview

Your reminder system currently uses only synthetic data. By integrating the Pitt Corpus (real dementia speech patterns), you'll get:

- **Better Real-World Performance**: Models trained on actual dementia speech patterns
- **Enhanced Feature Extraction**: Additional linguistic and cognitive markers
- **Improved Accuracy**: More robust predictions across different dementia severities
- **Clinical Validation**: Training data from validated clinical assessments

## 📁 What You Have

### Current Data Sources:
- ✅ **Synthetic Data**: `data/synthetic_reminder_data.csv` (2,000 samples)
- ✅ **Pitt Corpus**: `data/Pitt/` directory with real dementia speech data
  - Control group: `data/Pitt/Control/` (healthy participants)
  - Dementia group: `data/Pitt/Dementia/` (dementia patients)
  - Tasks: cookie theft, fluency, recall, sentence completion

### Current Models (Synthetic Only):
- Confusion Detection: 98.6% accuracy
- Cognitive Risk Assessment
- Caregiver Alert Prediction
- Response Classification

## 🚀 Integration Process

### Step 1: Quick Integration (Recommended)

Run the automated integration script:

```powershell
python scripts/run_integration.py
```

This will:
1. Extract features from Pitt Corpus (.cha files)
2. Create a balanced dataset (synthetic + real data)
3. Validate data quality
4. Train enhanced models
5. Generate performance reports

### Step 2: Manual Integration (Advanced)

If you want more control over the process:

#### Extract Pitt Corpus Features
```powershell
python scripts/integrate_pitt_data.py --extract-pitt --pitt-features data/pitt_features.csv
```

#### Create Balanced Dataset
```powershell
python scripts/integrate_pitt_data.py --create-balanced --output data/enhanced_training_data.csv --balance-ratio 0.3
```

#### Validate Integration
```powershell
python scripts/integrate_pitt_data.py --validate --output data/enhanced_training_data.csv
```

#### Train Enhanced Models
```powershell
python scripts/train_enhanced_reminder_models.py
```

#### Test Enhanced Models
```powershell
python scripts/test_enhanced_models.py
```

## 📊 Expected Improvements

### Enhanced Features
The integration adds these real-world features:

**Pitt-Derived Features:**
- `pitt_dementia_markers`: Real dementia speech patterns
- `narrative_coherence`: Story-telling ability assessment
- `task_completion`: Ability to complete cognitive tasks
- `linguistic_complexity`: Language sophistication measures
- `error_patterns`: Common dementia-related speech errors

**Enhanced Cognitive Features:**
- `word_finding_difficulty`: Difficulty accessing words
- `circumlocution`: Speaking around forgotten words  
- `tangentiality`: Going off-topic tendencies
- `semantic_drift`: Loss of topic coherence
- `discourse_coherence`: Overall communication effectiveness

### Performance Expectations
- **Accuracy**: 85-95% (up from synthetic-only performance)
- **Real-World Validity**: Much higher due to clinical data
- **Robustness**: Better performance across different dementia severities
- **False Positive Reduction**: Fewer incorrect alerts

## 📋 File Structure After Integration

```
data/
├── synthetic_reminder_data.csv          # Original synthetic data
├── pitt_features.csv                    # Extracted Pitt features  
├── enhanced_training_data.csv           # Combined dataset
└── enhanced_training_data_validation.json  # Data quality report

models/reminder_system/
├── confusion_detection_model.joblib     # Enhanced confusion model
├── cognitive_risk_model.joblib          # Enhanced risk model  
├── caregiver_alert_model.joblib         # Enhanced alert model
├── response_classifier_model.joblib     # Enhanced classifier
├── enhanced_training_metadata.json     # Training details
└── test_report.json                     # Model performance report

scripts/
├── integrate_pitt_data.py               # Data integration utility
├── train_enhanced_reminder_models.py   # Enhanced training script
├── test_enhanced_models.py             # Model validation
└── run_integration.py                  # Automated runner
```

## 🔧 Configuration Options

### Balance Ratio
Control the mix of synthetic vs. real data:
```bash
--balance-ratio 0.3   # 30% Pitt, 70% synthetic (default)
--balance-ratio 0.5   # 50% Pitt, 50% synthetic
--balance-ratio 0.2   # 20% Pitt, 80% synthetic
```

### Feature Groups
The integration uses these feature groups:
- **Text Features**: Hesitation, incoherence, confidence
- **Temporal Features**: Response time, pause patterns
- **Cognitive Features**: Memory, confusion, semantic drift
- **Context Features**: Task type, time, priority
- **Pitt Features**: Real dementia markers

## 🧪 Testing Your Enhanced Models

### Test Cases Include:
- **Healthy Responses**: Clear, confident answers
- **Mild Confusion**: Slight hesitation, uncertainty
- **Moderate Confusion**: Memory issues, help requests
- **High Confusion**: Severe disorientation, inability to respond

### Validation Metrics:
- **Accuracy**: Overall correct predictions
- **Precision**: Correct positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Balance of precision and recall

## 🚨 Troubleshooting

### Common Issues:

**1. Pitt Data Not Found**
```
Error: Pitt directory not found at data/Pitt
```
**Solution**: Ensure you have the Pitt Corpus dataset in the correct location.

**2. Feature Extraction Errors**
```
Error: Failed to extract features from .cha files
```
**Solution**: Check file encoding and format. Some .cha files may be corrupted.

**3. Memory Issues**
```
Error: Out of memory during training
```
**Solution**: Reduce batch size or use a smaller balance ratio.

**4. Model Training Failures**
```
Error: Failed to train enhanced models
```
**Solution**: Check data quality report and ensure features are properly aligned.

### Debug Commands:

Check data quality:
```powershell
python scripts/integrate_pitt_data.py --validate --output data/enhanced_training_data.csv
```

Test specific models:
```powershell
python scripts/test_enhanced_models.py
```

## 📈 Performance Monitoring

After integration, monitor these metrics:

### Real-World Performance:
- **Accuracy**: Target >90% on test cases
- **Latency**: Response time <200ms
- **Memory**: RAM usage for inference
- **False Positives**: Unnecessary alerts
- **False Negatives**: Missed critical cases

### Clinical Validation:
- Compare with healthcare provider assessments
- Track user satisfaction and system reliability
- Monitor caregiver alert effectiveness

## 🎉 Success Indicators

Your integration is successful when you see:

✅ **Data Integration**: Combined dataset created without errors  
✅ **Model Training**: All models train successfully  
✅ **Performance**: Test accuracy >85%  
✅ **Validation**: Quality report shows no critical issues  
✅ **Real-World Testing**: Improved performance on actual user interactions  

## 📞 Next Steps

1. **Deploy Enhanced Models**: Replace your current models with enhanced versions
2. **A/B Testing**: Compare synthetic-only vs. enhanced model performance
3. **User Feedback**: Collect feedback on improved system performance
4. **Continuous Learning**: Regular retraining with new data
5. **Clinical Validation**: Work with healthcare providers to validate improvements

## 🔬 Research Applications

The enhanced models enable:
- **Clinical Decision Support**: More accurate cognitive assessments
- **Early Detection**: Better identification of cognitive decline
- **Personalized Care**: Tailored interventions based on speech patterns
- **Research Studies**: Use in dementia research and clinical trials

## 📚 References

- Pitt Corpus: Clinical dementia speech database
- Cookie Theft Task: Standard cognitive assessment
- Feature Engineering: Linguistic markers for dementia detection
- Machine Learning: Ensemble methods for clinical prediction

---

**Need Help?** Check the validation reports, test results, and log files for detailed information about your integration process.