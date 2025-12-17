"""
WebSocket Routes for Real-Time Reminder System

Provides WebSocket endpoints for:
- Real-time reminder delivery to users
- Live caregiver alert notifications
- Instant response processing
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from typing import Dict, Any, Optional
import logging
import json

from src.features.reminder_system.realtime_engine import RealTimeReminderEngine
from src.services.reminder_db_service import ReminderDatabaseService

logger = logging.getLogger(__name__)

# Create router for WebSocket endpoints
ws_router = APIRouter(prefix="/ws", tags=["websocket"])

# Global real-time engine instance
realtime_engine = RealTimeReminderEngine()


@ws_router.on_event("startup")
async def startup_realtime_engine():
    """Start the real-time engine when the application starts."""
    await realtime_engine.start_engine()
    logger.info("Real-time reminder engine started")


@ws_router.on_event("shutdown")
async def shutdown_realtime_engine():
    """Stop the real-time engine when the application shuts down."""
    await realtime_engine.stop_engine()
    logger.info("Real-time reminder engine stopped")


@ws_router.websocket("/user/{user_id}")
async def websocket_user_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for user real-time notifications.
    
    Handles:
    - Real-time reminder delivery
    - Response feedback
    - Status updates
    """
    try:
        await realtime_engine.connect_user(user_id, websocket)
        
        while True:
            try:
                # Wait for messages from the user
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await handle_user_message(user_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"User {user_id} disconnected normally")
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing user message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to process message"
                }))
    
    except Exception as e:
        logger.error(f"WebSocket connection error for user {user_id}: {e}")
    
    finally:
        await realtime_engine.disconnect_user(user_id)


@ws_router.websocket("/caregiver/{caregiver_id}")
async def websocket_caregiver_endpoint(websocket: WebSocket, caregiver_id: str):
    """
    WebSocket endpoint for caregiver real-time alerts.
    
    Handles:
    - Real-time alert notifications
    - Alert acknowledgments
    - System status updates
    """
    try:
        await realtime_engine.connect_caregiver(caregiver_id, websocket)
        
        while True:
            try:
                # Wait for messages from the caregiver
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await handle_caregiver_message(caregiver_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"Caregiver {caregiver_id} disconnected normally")
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing caregiver message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to process message"
                }))
    
    except Exception as e:
        logger.error(f"WebSocket connection error for caregiver {caregiver_id}: {e}")
    
    finally:
        await realtime_engine.disconnect_caregiver(caregiver_id)


# Message handlers

async def handle_user_message(user_id: str, message: Dict[str, Any]):
    """Handle incoming messages from users."""
    message_type = message.get("type")
    
    if message_type == "reminder_response":
        # User responding to a reminder
        await handle_reminder_response(user_id, message)
    
    elif message_type == "status_request":
        # User requesting status update
        await handle_status_request(user_id, message)
    
    elif message_type == "ping":
        # Keep-alive ping
        await realtime_engine._send_user_message(user_id, {
            "type": "pong",
            "timestamp": message.get("timestamp")
        })
    
    else:
        logger.warning(f"Unknown message type from user {user_id}: {message_type}")


async def handle_caregiver_message(caregiver_id: str, message: Dict[str, Any]):
    """Handle incoming messages from caregivers."""
    message_type = message.get("type")
    
    if message_type == "acknowledge_alert":
        # Caregiver acknowledging an alert
        await handle_alert_acknowledgment(caregiver_id, message)
    
    elif message_type == "resolve_alert":
        # Caregiver resolving an alert
        await handle_alert_resolution(caregiver_id, message)
    
    elif message_type == "request_user_status":
        # Caregiver requesting user status
        await handle_user_status_request(caregiver_id, message)
    
    elif message_type == "ping":
        # Keep-alive ping
        await realtime_engine._send_caregiver_message(caregiver_id, {
            "type": "pong",
            "timestamp": message.get("timestamp")
        })
    
    else:
        logger.warning(f"Unknown message type from caregiver {caregiver_id}: {message_type}")


async def handle_reminder_response(user_id: str, message: Dict[str, Any]):
    """Handle user response to a reminder."""
    try:
        reminder_id = message.get("reminder_id")
        response_text = message.get("response_text", "")
        response_time = message.get("response_time_seconds")
        
        if not reminder_id:
            await realtime_engine._send_user_message(user_id, {
                "type": "error",
                "message": "Missing reminder_id in response"
            })
            return
        
        # Process the response through the real-time engine
        result = await realtime_engine.process_user_response(
            user_id, reminder_id, response_text, response_time
        )
        
        # Send confirmation back to user
        await realtime_engine._send_user_message(user_id, {
            "type": "response_processed",
            "reminder_id": reminder_id,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error handling reminder response: {e}")
        await realtime_engine._send_user_message(user_id, {
            "type": "error",
            "message": "Failed to process reminder response"
        })


async def handle_status_request(user_id: str, message: Dict[str, Any]):
    """Handle user status request."""
    try:
        # Get user's current reminders and status
        db_service = ReminderDatabaseService()
        reminders = await db_service.get_user_reminders(user_id, limit=10)
        analytics = await db_service.get_reminder_analytics(user_id, days_back=7)
        
        status_response = {
            "type": "status_update",
            "user_id": user_id,
            "active_reminders": len(reminders),
            "recent_analytics": analytics,
            "connection_status": "connected"
        }
        
        await realtime_engine._send_user_message(user_id, status_response)
        
    except Exception as e:
        logger.error(f"Error handling status request: {e}")
        await realtime_engine._send_user_message(user_id, {
            "type": "error",
            "message": "Failed to retrieve status"
        })


async def handle_alert_acknowledgment(caregiver_id: str, message: Dict[str, Any]):
    """Handle caregiver alert acknowledgment."""
    try:
        alert_id = message.get("alert_id")
        
        if not alert_id:
            await realtime_engine._send_caregiver_message(caregiver_id, {
                "type": "error",
                "message": "Missing alert_id"
            })
            return
        
        db_service = ReminderDatabaseService()
        success = await db_service.acknowledge_alert(alert_id, caregiver_id)
        
        await realtime_engine._send_caregiver_message(caregiver_id, {
            "type": "alert_acknowledged",
            "alert_id": alert_id,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error handling alert acknowledgment: {e}")
        await realtime_engine._send_caregiver_message(caregiver_id, {
            "type": "error",
            "message": "Failed to acknowledge alert"
        })


async def handle_alert_resolution(caregiver_id: str, message: Dict[str, Any]):
    """Handle caregiver alert resolution."""
    try:
        alert_id = message.get("alert_id")
        resolution_notes = message.get("resolution_notes", "")
        
        if not alert_id:
            await realtime_engine._send_caregiver_message(caregiver_id, {
                "type": "error",
                "message": "Missing alert_id"
            })
            return
        
        db_service = ReminderDatabaseService()
        success = await db_service.resolve_alert(alert_id, caregiver_id, resolution_notes)
        
        await realtime_engine._send_caregiver_message(caregiver_id, {
            "type": "alert_resolved",
            "alert_id": alert_id,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error handling alert resolution: {e}")
        await realtime_engine._send_caregiver_message(caregiver_id, {
            "type": "error",
            "message": "Failed to resolve alert"
        })


async def handle_user_status_request(caregiver_id: str, message: Dict[str, Any]):
    """Handle caregiver request for user status."""
    try:
        user_id = message.get("user_id")
        
        if not user_id:
            await realtime_engine._send_caregiver_message(caregiver_id, {
                "type": "error",
                "message": "Missing user_id"
            })
            return
        
        db_service = ReminderDatabaseService()
        
        # Get comprehensive user status
        reminders = await db_service.get_user_reminders(user_id, limit=5)
        analytics = await db_service.get_reminder_analytics(user_id, days_back=7)
        behavior_pattern = await db_service.get_behavior_pattern(user_id)
        recent_interactions = await db_service.get_user_interactions(user_id, days_back=3)
        
        user_status = {
            "type": "user_status",
            "user_id": user_id,
            "connection_status": "connected" if user_id in realtime_engine.user_connections else "offline",
            "recent_reminders": reminders,
            "analytics": analytics,
            "behavior_pattern": behavior_pattern,
            "recent_interactions": recent_interactions[:10]  # Last 10 interactions
        }
        
        await realtime_engine._send_caregiver_message(caregiver_id, user_status)
        
    except Exception as e:
        logger.error(f"Error handling user status request: {e}")
        await realtime_engine._send_caregiver_message(caregiver_id, {
            "type": "error",
            "message": "Failed to retrieve user status"
        })


# Additional REST endpoints for WebSocket management

@ws_router.get("/status")
async def get_websocket_status():
    """Get current WebSocket connection status."""
    return {
        "active_user_connections": len(realtime_engine.user_connections),
        "active_caregiver_connections": len(realtime_engine.caregiver_connections),
        "engine_running": realtime_engine.is_running,
        "connected_users": list(realtime_engine.user_connections.keys()),
        "connected_caregivers": list(realtime_engine.caregiver_connections.keys())
    }


@ws_router.post("/broadcast/users")
async def broadcast_to_users(message: Dict[str, Any]):
    """Broadcast a message to all connected users."""
    sent_count = 0
    for user_id in realtime_engine.user_connections.keys():
        success = await realtime_engine._send_user_message(user_id, message)
        if success:
            sent_count += 1
    
    return {
        "message_sent_to": sent_count,
        "total_connections": len(realtime_engine.user_connections)
    }


@ws_router.post("/broadcast/caregivers")
async def broadcast_to_caregivers(message: Dict[str, Any]):
    """Broadcast a message to all connected caregivers."""
    sent_count = 0
    for caregiver_id in realtime_engine.caregiver_connections.keys():
        success = await realtime_engine._send_caregiver_message(caregiver_id, message)
        if success:
            sent_count += 1
    
    return {
        "message_sent_to": sent_count,
        "total_connections": len(realtime_engine.caregiver_connections)
    }