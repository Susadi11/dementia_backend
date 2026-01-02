"""
Simple FastAPI app with Smart Reminder System only

This version focuses on the Context-Aware Smart Reminder System
without the torch-dependent AI features.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import sys

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import enhanced Swagger configuration
from swagger_config_fixed import create_enhanced_openapi_schema, setup_swagger_ui_config
from swagger_testing import testing_router

# Import only reminder routes (without torch dependencies)
from src.routes import healthcheck, reminder_routes_complete

# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="Context-Aware Smart Reminder System",
    description="Comprehensive API for intelligent reminder management and caregiver integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (only reminder system routes)
app.include_router(healthcheck.router)
app.include_router(reminder_routes_complete.router)
app.include_router(testing_router)

# Setup enhanced Swagger configuration
setup_swagger_ui_config(app)

# Custom OpenAPI schema
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi():
    """Get enhanced OpenAPI schema."""
    return create_enhanced_openapi_schema(app)

@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint providing system information.
    
    Returns basic information about the Context-Aware Smart Reminder System.
    """
    return {
        "name": "Context-Aware Smart Reminder System",
        "version": "2.0.0",
        "description": "Intelligent reminder management with caregiver integration",
        "features": [
            "Smart reminder scheduling",
            "Context-aware notifications",
            "Caregiver alerts and escalation",
            "User interaction tracking",
            "Behavioral analytics",
            "Multi-modal reminder support"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "reminders": "/api/reminders",
            "testing": "/api/testing"
        }
    }

def main():
    """Main function to start the reminder system API server."""
    print("\n" + "="*80)
    print("📅 Context-Aware Smart Reminder System API")
    print("="*80)
    print("\n📚 API Documentation:")
    print("  • Swagger UI: http://localhost:8000/docs")
    print("  • ReDoc:      http://localhost:8000/redoc")
    print("\n🔔 Smart Reminder Endpoints:")
    print("  • Create Reminder:     POST http://localhost:8000/api/reminders/")
    print("  • Get User Reminders:  GET  http://localhost:8000/api/reminders/user/{user_id}")
    print("  • Update Reminder:     PUT  http://localhost:8000/api/reminders/{reminder_id}")
    print("  • Complete Reminder:   POST http://localhost:8000/api/reminders/{reminder_id}/complete")
    print("  • Get Due Reminders:   GET  http://localhost:8000/api/reminders/due/now")
    print("  • User Statistics:     GET  http://localhost:8000/api/reminders/stats/user/{user_id}")
    print("\n🧪 Testing Endpoints:")
    print("  • Generate Test Data:  POST http://localhost:8000/api/testing/scenarios/{type}")
    print("  • Get Examples:        GET  http://localhost:8000/api/testing/swagger-examples")
    print("  • Testing Health:      GET  http://localhost:8000/api/testing/health-check")
    print("\n🔧 System Endpoints:")
    print("  • System Info:    GET  http://localhost:8000/")
    print("  • Health Check:   GET  http://localhost:8000/health")
    print("\n📱 Features:")
    print("  • Context-aware reminder scheduling")
    print("  • Caregiver integration and alerts")
    print("  • User interaction tracking")
    print("  • Behavioral analytics")
    print("  • Multi-priority escalation system")
    print("\n" + "="*80)
    print("💡 Tip: Press CTRL+C to stop the server")
    print("="*80 + "\n")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n📅 Smart Reminder System server stopped by user")
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        print("\nMake sure you have:")
        print("  1. Installed FastAPI requirements: pip install fastapi uvicorn")
        print("  2. MongoDB connection configured in src/database.py")
        print("  3. Required dependencies for reminder system")
        sys.exit(1)

if __name__ == "__main__":
    main()