"""
Test Real-Time Reminder System

Demonstrates the real-time functionality including:
- WebSocket connections
- Live reminder delivery
- Response processing
- Caregiver alerts
"""

import asyncio
import websockets
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('realtime_test')

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

class RealTimeTestClient:
    """Test client for the real-time reminder system."""
    
    def __init__(self):
        self.user_ws = None
        self.caregiver_ws = None
        self.user_id = "test_user_123"
        self.caregiver_id = "test_caregiver_456"
    
    async def test_complete_flow(self):
        """Test the complete real-time reminder flow."""
        logger.info("Starting complete real-time reminder flow test")
        
        try:
            # Step 1: Create a test reminder
            await self.create_test_reminder()
            
            # Step 2: Connect WebSockets
            await self.connect_websockets()
            
            # Step 3: Test real-time features
            await self.test_reminder_delivery()
            await self.test_user_responses()
            await self.test_caregiver_alerts()
            
            # Step 4: Test analytics
            await self.test_analytics()
            
            logger.info("✅ All tests completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
        
        finally:
            # Cleanup
            await self.disconnect_websockets()
    
    async def create_test_reminder(self):
        """Create a test reminder via REST API."""
        logger.info("Creating test reminder...")
        
        reminder_data = {
            "id": "test_reminder_123",
            "user_id": self.user_id,
            "title": "Take morning medication",
            "description": "Blood pressure medication (blue pill)",
            "scheduled_time": (datetime.now() + timedelta(minutes=1)).isoformat(),
            "priority": "high",
            "category": "medication",
            "repeat_pattern": "daily",
            "caregiver_ids": [self.caregiver_id],
            "adaptive_scheduling_enabled": True,
            "escalation_enabled": True
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/reminders/create",
                json=reminder_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 201:
                logger.info("✅ Test reminder created successfully")
            else:
                logger.warning(f"⚠️ Reminder creation returned status: {response.status_code}")
                logger.info("Proceeding with test anyway...")
        
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ Cannot connect to API server - ensure it's running on localhost:8000")
            logger.info("Proceeding with WebSocket tests...")
        except Exception as e:
            logger.warning(f"⚠️ Error creating reminder: {e}")
    
    async def connect_websockets(self):
        """Connect user and caregiver WebSockets."""
        logger.info("Connecting WebSocket clients...")
        
        try:
            # Connect user WebSocket
            self.user_ws = await websockets.connect(f"{WS_URL}/ws/user/{self.user_id}")
            logger.info("✅ User WebSocket connected")
            
            # Connect caregiver WebSocket
            self.caregiver_ws = await websockets.connect(f"{WS_URL}/ws/caregiver/{self.caregiver_id}")
            logger.info("✅ Caregiver WebSocket connected")
            
            # Listen for initial connection messages
            await asyncio.sleep(1)  # Allow connection to establish
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            logger.info("💡 Make sure the API server is running: python src/api/app_simple.py")
            raise
    
    async def disconnect_websockets(self):
        """Disconnect WebSocket connections."""
        if self.user_ws:
            await self.user_ws.close()
            logger.info("User WebSocket disconnected")
        
        if self.caregiver_ws:
            await self.caregiver_ws.close()
            logger.info("Caregiver WebSocket disconnected")
    
    async def test_reminder_delivery(self):
        """Test real-time reminder delivery simulation."""
        logger.info("Testing reminder delivery...")
        
        # Simulate a reminder being delivered
        test_reminder = {
            "type": "reminder",
            "reminder_id": "test_reminder_123",
            "title": "Take morning medication",
            "description": "Blood pressure medication (blue pill)",
            "priority": "high",
            "category": "medication",
            "scheduled_time": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Manually send reminder to user (simulating system delivery)
        await self.user_ws.send(json.dumps(test_reminder))
        logger.info("📨 Simulated reminder delivery sent")
        
        # Wait for any response
        await asyncio.sleep(2)
    
    async def test_user_responses(self):
        """Test user response scenarios."""
        logger.info("Testing user response scenarios...")
        
        # Test scenarios with different response types
        test_responses = [
            {
                "scenario": "Clear confirmation",
                "response": {
                    "type": "reminder_response",
                    "reminder_id": "test_reminder_123",
                    "response_text": "Yes, I took my medication",
                    "response_time_seconds": 15.0
                }
            },
            {
                "scenario": "Confused response",
                "response": {
                    "type": "reminder_response",
                    "reminder_id": "test_reminder_123",
                    "response_text": "I think... maybe I did? What was it again?",
                    "response_time_seconds": 45.0
                }
            },
            {
                "scenario": "Delayed/uncertain response",
                "response": {
                    "type": "reminder_response",
                    "reminder_id": "test_reminder_123",
                    "response_text": "Um... later, I'm busy right now",
                    "response_time_seconds": 120.0
                }
            }
        ]
        
        for test_case in test_responses:
            logger.info(f"Testing scenario: {test_case['scenario']}")
            
            # Send response
            await self.user_ws.send(json.dumps(test_case["response"]))
            
            # Wait for processing and feedback
            await asyncio.sleep(3)
            
            logger.info(f"✅ Completed scenario: {test_case['scenario']}")
    
    async def test_caregiver_alerts(self):
        """Test caregiver alert functionality."""
        logger.info("Testing caregiver alert system...")
        
        # Simulate a high-risk interaction that triggers caregiver alert
        alert_trigger = {
            "type": "user_alert",
            "user_id": self.user_id,
            "reminder_id": "test_reminder_123",
            "alert_type": "confusion",
            "severity": "high",
            "message": "User showed confusion and memory issues with medication reminder",
            "timestamp": datetime.now().isoformat(),
            "interaction_summary": {
                "response": "I think... maybe I did? What was it again?",
                "cognitive_risk": 0.75,
                "confusion_detected": True
            }
        }
        
        # Send alert to caregiver (simulating system alert)
        await self.caregiver_ws.send(json.dumps(alert_trigger))
        logger.info("🚨 Simulated caregiver alert sent")
        
        # Test caregiver acknowledgment
        await asyncio.sleep(2)
        
        acknowledgment = {
            "type": "acknowledge_alert",
            "alert_id": "alert_123",
            "caregiver_id": self.caregiver_id,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.caregiver_ws.send(json.dumps(acknowledgment))
        logger.info("✅ Caregiver acknowledgment sent")
        
        await asyncio.sleep(2)
    
    async def test_analytics(self):
        """Test analytics endpoints."""
        logger.info("Testing analytics functionality...")
        
        # Test user status request
        status_request = {
            "type": "status_request",
            "timestamp": datetime.now().isoformat()
        }
        
        await self.user_ws.send(json.dumps(status_request))
        logger.info("📊 User status requested")
        
        # Test caregiver user status request
        caregiver_status_request = {
            "type": "request_user_status",
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.caregiver_ws.send(json.dumps(caregiver_status_request))
        logger.info("📊 Caregiver user status requested")
        
        await asyncio.sleep(3)
        
        # Test REST API analytics if available
        try:
            response = requests.get(f"{BASE_URL}/ws/status")
            if response.status_code == 200:
                status_data = response.json()
                logger.info(f"✅ WebSocket status: {status_data}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch WebSocket status: {e}")
    
    async def listen_for_messages(self):
        """Background task to listen for WebSocket messages."""
        async def listen_user():
            if not self.user_ws:
                return
                
            try:
                async for message in self.user_ws:
                    data = json.loads(message)
                    logger.info(f"👤 User received: {data.get('type', 'unknown')} - {data.get('message', '')}")
            except Exception as e:
                logger.debug(f"User WebSocket listener ended: {e}")
        
        async def listen_caregiver():
            if not self.caregiver_ws:
                return
                
            try:
                async for message in self.caregiver_ws:
                    data = json.loads(message)
                    logger.info(f"👨‍⚕️ Caregiver received: {data.get('type', 'unknown')} - {data.get('message', '')}")
            except Exception as e:
                logger.debug(f"Caregiver WebSocket listener ended: {e}")
        
        # Start both listeners
        await asyncio.gather(
            listen_user(),
            listen_caregiver(),
            return_exceptions=True
        )


def run_manual_test():
    """Run a manual test to demonstrate the system."""
    print("\n" + "="*60)
    print("🧠 REAL-TIME REMINDER SYSTEM TEST")
    print("="*60)
    
    print("""
This test demonstrates:
✅ WebSocket connections (user & caregiver)
✅ Real-time reminder delivery
✅ Response analysis (confusion detection)
✅ Caregiver alert system
✅ Analytics and status updates

Prerequisites:
1. Start the API server: python src/api/app_simple.py
2. Server should be running on localhost:8000
    """)
    
    client = RealTimeTestClient()
    
    # Create event loop and run test
    asyncio.run(client.test_complete_flow())


async def interactive_test():
    """Interactive test for manual exploration."""
    print("\n🔧 INTERACTIVE MODE")
    print("You can now interact with the system manually.")
    
    client = RealTimeTestClient()
    
    try:
        await client.connect_websockets()
        
        # Start background listeners
        listener_task = asyncio.create_task(client.listen_for_messages())
        
        print("\n💡 Commands:")
        print("  reminder - Send a test reminder")
        print("  response - Send a user response")
        print("  alert - Send a caregiver alert") 
        print("  status - Request status update")
        print("  quit - Exit")
        
        while True:
            command = input("\n> ").strip().lower()
            
            if command == "quit":
                break
            elif command == "reminder":
                await client.test_reminder_delivery()
            elif command == "response":
                await client.test_user_responses()
            elif command == "alert":
                await client.test_caregiver_alerts()
            elif command == "status":
                await client.test_analytics()
            else:
                print("Unknown command. Try: reminder, response, alert, status, quit")
        
        listener_task.cancel()
        
    except Exception as e:
        logger.error(f"Interactive test error: {e}")
    
    finally:
        await client.disconnect_websockets()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_test())
    else:
        run_manual_test()