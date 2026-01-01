"""
MongoDB Database Connection
Handles connection to MongoDB Atlas cluster
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
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

            # Create MongoDB client
            cls.client = AsyncIOMotorClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )

            # Test the connection
            await cls.client.admin.command('ping')

            # Get database
            cls.db = cls.client[db_name]

            logger.info(f"Successfully connected to MongoDB database: {db_name}")

            return True

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
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
        if not cls.db:
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
