import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    try:
        r = requests.get(f"{BASE_URL}/health")
        print(f"Health: {r.status_code}")
        print(r.json())
        return r.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_motor_baseline():
    print("\nTesting Motor Baseline...")
    # 1. Calibrate
    caldata = {
        "userId": "test_verification_001",
        "tapTimes": [0.25, 0.26, 0.24, 0.25, 0.27]
    }
    r = requests.post(f"{BASE_URL}/game/calibration", json=caldata)
    print(f"Calibration POST: {r.status_code}")
    if r.status_code != 200:
        print(r.text)

    # 2. Get Baseline
    r = requests.get(f"{BASE_URL}/game/motor-baseline/test_verification_001")
    print(f"Baseline GET: {r.status_code}")
    print(r.json())

def test_game_session():
    print("\nTesting Game Session...")
    session_data = {
        "userId": "test_verification_001",
        "sessionId": f"sess_{int(time.time())}",
        "gameType": "card_matching",
        "level": 1,
        "trials": [
            {"rt_raw": 1.5, "correct": 1},
            {"rt_raw": 1.6, "correct": 1},
            {"rt_raw": 2.2, "correct": 0}
        ]
    }
    r = requests.post(f"{BASE_URL}/game/session", json=session_data)
    print(f"Session POST: {r.status_code}")
    if r.status_code == 201:
        print("Session created successfully")
        print(r.json())
    else:
        print(r.text)

def test_risk_predict():
    print("\nTesting Risk Prediction...")
    # Need to wait/submit enough sessions?
    # Or just call it with N=1 (might degrade gracefully)
    r = requests.post(f"{BASE_URL}/risk/predict/test_verification_001?N=5")
    print(f"Risk Predict POST: {r.status_code}")
    if r.status_code == 200:
        print(r.json())
    else:
        print(r.text)

def test_risk_history():
    print("\nTesting Risk History...")
    r = requests.get(f"{BASE_URL}/risk/history/test_verification_001")
    print(f"Risk History GET: {r.status_code}")
    print(r.json())

if __name__ == "__main__":
    print("Waiting for server to be ready...")
    
    server_ready = False
    for i in range(12):  # Try for 60 seconds
        if test_health():
            server_ready = True
            break
        print(f"Server not ready, retrying in 5s... ({i+1}/12)")
        time.sleep(5)
    
    if server_ready:
        try:
            test_motor_baseline()
            test_game_session()
            test_risk_predict()
            test_risk_history()
        except Exception as e:
            print(f"Tests failed: {e}")
    else:
        print("Server failed to come online after 60s")
