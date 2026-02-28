"""
MongoDB Database Connection
Handles connection to MongoDB Atlas cluster
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class Database:
    """MongoDB Database Manager"""

    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect_to_database(cls):
        """
        Connect to MongoDB database.
        Called on application startup.
        """
        try:
            # Get MongoDB URI from environment
            mongodb_uri = os.getenv("MONGODB_URI")
            db_name = os.getenv("MONGODB_DB_NAME", "dementia_care_db")

            if not mongodb_uri:
                logger.error("MONGODB_URI not found in environment variables")
                raise ValueError("MONGODB_URI is required")

            logger.info("Connecting to MongoDB...")
            print(f"Connecting to MongoDB Atlas ({db_name})...")

            # Create MongoDB client
            cls.client = AsyncIOMotorClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )

            # Test the connection (timeout guards against asyncio.CancelledError on slow Atlas)
            await asyncio.wait_for(cls.client.admin.command('ping'), timeout=15)

            # Get database
            cls.db = cls.client[db_name]

            logger.info(f"Successfully connected to MongoDB database: {db_name}")
            print(f"✅ PASSED: Connected to MongoDB ({db_name})")

            return True

        except asyncio.CancelledError:
            logger.error("MongoDB ping was cancelled (server startup interrupted)")
            print("❌ ERROR: MongoDB connection ping was cancelled")
            raise
        except asyncio.TimeoutError:
            logger.error("MongoDB connection timed out after 15 seconds")
            print("❌ FAILED: MongoDB connection timed out")
            raise ConnectionFailure("MongoDB ping timed out")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            print(f"❌ FAILED: Could not connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            print(f"❌ ERROR: Unexpected error connecting to DB: {e}")
            raise

    @classmethod
    async def close_database_connection(cls):
        """
        Close MongoDB connection.
        Called on application shutdown.
        """
        if cls.client:
            logger.info("Closing MongoDB connection...")
            cls.client.close()
            logger.info("MongoDB connection closed")

    @classmethod
    def get_collection(cls, collection_name: str):
        """
        Get a MongoDB collection.

        Args:
            collection_name: Name of the collection

        Returns:
            MongoDB collection object
        """
        if cls.db is None:
            raise RuntimeError("Database not connected. Call connect_to_database() first.")

        return cls.db[collection_name]

    @classmethod
    async def create_indexes(cls):
        """
        Create database indexes for better performance.
        Called after successful connection.
        """
        try:
            logger.info("Creating database indexes...")

            # Example: Create index on users collection
            users = cls.get_collection("users")
            await users.create_index("user_id", unique=True)
            await users.create_index("created_at")

            # Example: Create index on chat sessions
            sessions = cls.get_collection("chat_sessions")
            await sessions.create_index("session_id", unique=True)
            await sessions.create_index("user_id")
            await sessions.create_index("created_at")

            # Example: Create index on chat messages
            messages = cls.get_collection("chat_messages")
            await messages.create_index("session_id")
            await messages.create_index("timestamp")

            # Reminder system indexes
            reminders = cls.get_collection("reminders")
            await reminders.create_index("user_id")
            await reminders.create_index("scheduled_time")
            await reminders.create_index("status")
            await reminders.create_index([("user_id", 1), ("status", 1)])
            
            # Interaction indexes
            interactions = cls.get_collection("reminder_interactions")
            await interactions.create_index("user_id")
            await interactions.create_index("reminder_id")
            await interactions.create_index("interaction_time")
            
            # Behavior pattern indexes
            patterns = cls.get_collection("user_behavior_patterns")
            await patterns.create_index("user_id")
            await patterns.create_index("category")
            
            # Caregiver alert indexes
            alerts = cls.get_collection("caregiver_alerts")
            await alerts.create_index("caregiver_id")
            await alerts.create_index("created_at")

            # Chat detection session indexes (12-parameter system)
            chat_detection_sessions = cls.get_collection("chat_detection_sessions")
            await chat_detection_sessions.create_index("session_id", unique=True)
            await chat_detection_sessions.create_index("user_id")
            await chat_detection_sessions.create_index("timestamp")
            await chat_detection_sessions.create_index([("user_id", 1), ("timestamp", -1)])
            await chat_detection_sessions.create_index("time_window")
            await chat_detection_sessions.create_index("date")

            logger.info("Database indexes created successfully")

        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")


# Convenience functions
async def get_database():
    """Get database instance"""
    return Database.db


def get_collection(collection_name: str):
    """Get a collection by name"""
    return Database.get_collection(collection_name)
