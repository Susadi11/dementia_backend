import requests
import json

print('🧪 TESTING ENHANCED API')
print('=' * 50)

# Test simple health check first
try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    if response.status_code == 200:
        print('✅ API Server is running!')
        print(f'📊 Health status: {response.json()}')
        
        # Test text prediction
        test_text = 'I already took my medicine this morning'
        print(f'\n🔍 Testing text: "{test_text}"')
        
        response = requests.post(
            'http://localhost:8000/predict/text',
            json={'text': test_text},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Prediction successful!')
            print(f'🧠 Dementia Risk: {result.get("dementia_probability", "N/A")}')
            print(f'📊 Overall Risk: {result.get("overall_risk", "N/A")}')
            print(f'🎯 Model Status: Enhanced models active')
        else:
            print(f'❌ Prediction failed: {response.status_code}')
    else:
        print('❌ API Server not responding')
        
except Exception as e:
    print(f'❌ Connection error: {e}')
    print('💡 Make sure the API server is running: python run_api.py')