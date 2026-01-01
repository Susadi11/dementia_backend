"""
Enhanced Swagger/OpenAPI Configuration for Context-Aware Smart Reminder System

This module provides comprehensive OpenAPI documentation and testing capabilities
for the smart reminder system.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any

def create_enhanced_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Create enhanced OpenAPI schema with comprehensive documentation.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Enhanced OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Context-Aware Smart Reminder System",
        version="2.0.0",
        description="""
        ## Context-Aware Smart Reminder System API

        **Intelligent reminder management for enhanced care and medication adherence**

        ### 📅 **Smart Reminder Features**
        - **Context-Aware Scheduling**: Intelligent reminder timing based on user behavior patterns
        - **Multi-Priority System**: Critical, high, medium, and low priority classifications
        - **Category Management**: Medication, appointment, meal, exercise, and social reminders
        - **Repeat Patterns**: Daily, weekly, monthly, or custom recurring schedules

        ### 👥 **Caregiver Integration**
        - **Real-time Alerts**: Instant notifications when reminders are missed
        - **Escalation Management**: Automatic escalation based on configurable thresholds
        - **Multi-Caregiver Support**: Connect multiple caregivers to each user
        - **Override Capabilities**: Caregivers can modify or dismiss reminders when appropriate

        ### 📊 **User Interaction Tracking**
        - **Response Analysis**: Track confirmed, missed, and delayed reminder responses
        - **Behavioral Patterns**: Long-term analysis of user compliance and behavior
        - **Adaptive Scheduling**: Reminder timing adapts based on user response patterns
        - **Voice & Text Support**: Multiple interaction modalities for accessibility

        ### 📈 **Analytics & Monitoring**
        - **Compliance Reporting**: Detailed statistics on reminder adherence
        - **Risk Assessment**: Identify patterns that may indicate cognitive changes
        - **Performance Metrics**: Success rates, response times, and escalation frequencies
        - **Trend Analysis**: Long-term behavioral and health indicator tracking

        ### 🔐 **Security & Privacy**
        - **HIPAA-Compliant**: Healthcare data protection standards
        - **Encrypted Storage**: All personal data encrypted at rest and in transit  
        - **Audit Logging**: Complete audit trail of all system interactions
        - **User Consent**: Granular privacy controls and consent management

        ---
        
        **Base URL**: `http://localhost:8000`
        
        **Authentication**: API key authentication (header: `X-API-Key`)
        
        **Rate Limiting**: 1000 requests per hour per user
        
        **Support**: For integration support, see the testing endpoints
        """,
        routes=app.routes,
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ]
    )

    # Enhanced schema with additional metadata
    openapi_schema["info"]["contact"] = {
        "name": "Smart Reminder API Team",
        "email": "api-support@reminder-system.example.com"
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }

    # Add comprehensive tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Info",
            "description": "System information and overview"
        },
        {
            "name": "Health Check", 
            "description": "System health and status monitoring"
        },
        {
            "name": "Smart Reminders",
            "description": "Context-aware reminder creation, management, and scheduling"
        },
        {
            "name": "User Interactions",
            "description": "Track and analyze user responses to reminders"
        },
        {
            "name": "Caregiver Portal",
            "description": "Caregiver dashboard, notifications, and management tools"
        },
        {
            "name": "Analytics & Reports", 
            "description": "Statistics, compliance reporting, and behavioral analytics"
        },
        {
            "name": "Testing & Demo",
            "description": "Testing utilities and example data generation"
        }
    ]

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

def setup_swagger_ui_config(app: FastAPI) -> None:
    """
    Configure Swagger UI with enhanced settings for better testing experience.
    
    Args:
        app: FastAPI application instance
    """
    
    # Enhanced Swagger UI configuration
    app.swagger_ui_parameters = {
        "deepLinking": True,
        "displayRequestDuration": True,
        "docExpansion": "none",
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "tryItOutEnabled": True
    }