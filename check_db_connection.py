import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_check")

# Load environment variables
load_dotenv()

async def check_connection():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME")
    
    print("-" * 50)
    print(f"Checking MongoDB Connection...")
    print(f"URI found: {'Yes' if uri else 'No'}")
    # Hide password for security in logs
    if uri:
        safe_uri = uri.split('@')[-1] if '@' in uri else "..."
        print(f"URI Host: ...@{safe_uri}")
    print(f"Target DB: {db_name}")
    print("-" * 50)

    if not uri:
        print("ERROR: MONGODB_URI is missing in .env")
        return

    try:
        print("Attempting to connect (timeout=5s)...")
        client = AsyncIOMotorClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        
        # Force a command to actually test connection
        print("Pinging server...")
        await client.admin.command('ping')
        print("✅ SUCCESS: Connected to MongoDB Atlas!")
        
        # Check database access
        db = client[db_name]
        print(f"Database '{db_name}' selected.")
        collections = await db.list_collection_names()
        print(f"Collections found: {len(collections)}")
        print(f"List: {collections}")
        
    except Exception as e:
        print("\n❌ CONNECTION FAILED")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("\nTroubleshooting tips:")
        print("1. IP Whitelist: Go to MongoDB Atlas > Network Access > Add IP Address > 'Allow Access from Anywhere' (temporarily)")
        print("2. Credentials: Check username/password in URI")
        print("3. Firewall: Ensure port 27017 is not blocked")

if __name__ == "__main__":
    asyncio.run(check_connection())
