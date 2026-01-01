# Swagger Testing Guide for Context-Aware Smart Reminder System

## 🚀 Quick Start with Swagger UI

### Accessing Swagger Interface

1. **Start your FastAPI server:**
   ```bash
   python run_enhanced_api.py
   ```

2. **Open Swagger UI:**
   - **Interactive Docs**: http://localhost:8000/docs
   - **ReDoc**: http://localhost:8000/redoc
   - **OpenAPI Schema**: http://localhost:8000/openapi.json

### 🧪 Testing Workflow

## 1. System Health Check
Start by testing these endpoints to ensure everything is working:

- `GET /health` - Basic API health
- `GET /api/testing/health-check` - Testing system health
- `GET /api/reminders/health` - Reminder system health

## 2. Get Testing Examples
Use this endpoint to get ready-to-use test data:

- `GET /api/testing/swagger-examples` - Get comprehensive examples

## 3. Generate Test Scenarios
Create realistic test data for different cognitive levels:

- `POST /api/testing/scenarios/{scenario_type}`
  - `mild_cognitive_impairment`
  - `moderate_dementia` 
  - `normal_cognitive_function`
  - `mixed_symptoms`

## 📋 Step-by-Step Testing Guide

### Phase 1: Basic Reminder Testing

1. **Create a Reminder**
   ```
   POST /api/reminders/
   ```
   Use this sample JSON:
   ```json
   {
     "user_id": "test_user_123",
     "title": "Take morning medication",
     "description": "Blood pressure medication - 2 blue pills with water",
     "scheduled_time": "2025-12-27T08:00:00",
     "priority": "high",
     "category": "medication",
     "repeat_pattern": "daily",
     "caregiver_ids": ["caregiver_456"],
     "notify_caregiver_on_miss": true,
     "escalation_threshold_minutes": 30
   }
   ```

2. **Get User Reminders**
   ```
   GET /api/reminders/user/{user_id}
   ```
   Use: `test_user_123`

3. **Get Due Reminders**
   ```
   GET /api/reminders/due/now
   ```

### Phase 2: Interaction Testing

4. **Record User Interaction**
   ```
   POST /api/reminders/interactions
   ```
   Sample JSON:
   ```json
   {
     "reminder_id": "your_reminder_id_here",
     "user_id": "test_user_123",
     "interaction_type": "confirmed",
     "user_response_text": "Yes, I took my medication as prescribed"
   }
   ```

5. **Complete a Reminder**
   ```
   POST /api/reminders/{reminder_id}/complete
   ```

### Phase 3: Analytics Testing

6. **Get User Statistics**
   ```
   GET /api/reminders/stats/user/{user_id}
   ```

7. **Update Reminder**
   ```
   PUT /api/reminders/{reminder_id}
   ```

### Phase 4: AI Analysis Testing

8. **Analyze Text**
   ```
   POST /api/analysis/text
   ```
   Sample JSON:
   ```json
   {
     "text": "I sometimes forget where I put my keys and have trouble remembering appointments"
   }
   ```

9. **Session Analysis**
   ```
   POST /api/analysis/session
   ```

## 🎯 Advanced Testing Scenarios

### Scenario 1: Medication Adherence Flow
1. Create medication reminder
2. Simulate user missing reminder (no response)
3. Check caregiver notifications
4. Record delayed compliance
5. Analyze adherence patterns

### Scenario 2: Cognitive Assessment Flow
1. Generate test scenario for mild cognitive impairment
2. Analyze sample conversations 
3. Create adaptive reminders based on cognitive state
4. Monitor interaction patterns
5. Generate risk assessment reports

### Scenario 3: Caregiver Integration Flow
1. Create high-priority reminder with caregiver alerts
2. Simulate missed reminder
3. Test escalation thresholds
4. Verify caregiver notification system
5. Check override capabilities

## 🔧 Swagger UI Features for Testing

### Interactive Features
- **Try It Out**: Test endpoints directly in the browser
- **Request/Response Examples**: Pre-filled realistic data
- **Schema Validation**: Real-time validation of request bodies
- **Response Inspection**: Detailed response analysis

### Authentication Testing
- **API Key**: Use header `X-API-Key: your-test-key`
- **Bearer Token**: Future JWT implementation ready

### Batch Testing
- **Multiple Scenarios**: Use testing endpoints to create bulk data
- **Sequential Testing**: Chain API calls using response data
- **Data Persistence**: Test data persists across sessions

## 📊 Monitoring Test Results

### Key Metrics to Observe
- **Response Times**: API performance under different loads
- **Success Rates**: Percentage of successful operations
- **Error Patterns**: Common failure points
- **Data Consistency**: Database state after operations

### Swagger UI Analytics
- **Request Duration**: Built-in timing display
- **Response Size**: Payload optimization insights
- **Status Codes**: HTTP status distribution
- **Error Messages**: Detailed error information

## 🛠️ Customization Options

### Swagger UI Configuration
The system includes enhanced Swagger UI with:
- **Deep Linking**: Direct links to specific operations
- **Request/Response Interceptors**: Custom logging
- **Filter Support**: Search through operations
- **Expanded Documentation**: Comprehensive descriptions

### Environment Switching
Test against different environments:
- **Development**: http://localhost:8000
- **Testing**: http://localhost:8001
- **Staging**: https://staging-api.dementia-care.example.com

## 🚨 Common Testing Pitfalls

### Avoid These Mistakes
1. **Not checking system health first**: Always start with health endpoints
2. **Using invalid timestamp formats**: Use ISO 8601 format
3. **Missing required fields**: Check schema requirements
4. **Not testing error scenarios**: Test both success and failure cases
5. **Ignoring response data**: Use response IDs in subsequent calls

### Best Practices
1. **Use consistent test user IDs**: Makes data tracking easier
2. **Clean up test data**: Remove test reminders after testing
3. **Test edge cases**: Invalid inputs, missing fields, etc.
4. **Monitor logs**: Check server logs for detailed error information
5. **Test sequentially**: Some operations depend on previous results

## 📈 Performance Testing

### Load Testing with Swagger
1. **Baseline Testing**: Test single operations
2. **Concurrent Users**: Simulate multiple users
3. **Data Volume**: Test with large datasets
4. **Long-Running Sessions**: Extended conversation analysis

### Monitoring Performance
- **Response Time Trends**: Track API performance over time
- **Memory Usage**: Monitor server resource consumption
- **Database Performance**: Check query execution times
- **Error Rate Patterns**: Identify performance bottlenecks

## 🔍 Debugging with Swagger

### Troubleshooting Guide
1. **Check OpenAPI Schema**: Validate against /openapi.json
2. **Inspect Network Requests**: Use browser dev tools
3. **Validate Request Format**: Ensure JSON structure is correct
4. **Check Response Headers**: Look for error details
5. **Review Server Logs**: Check application logs for details

This comprehensive testing approach ensures your Context-Aware Smart Reminder System works correctly across all scenarios and provides reliable dementia detection and care management capabilities.