import asyncio
from src.database import Database

USER_ID = "USER-ASHI-20-E918"


async def main() -> None:
    await Database.connect_to_database()
    col = Database.get_collection("reminders")

    docs = []
    async for d in col.find({"user_id": USER_ID}).sort("scheduled_time", -1).limit(20):
        docs.append(
            {
                "id": str(d.get("_id")),
                "title": d.get("title"),
                "category": d.get("category"),
                "status": d.get("status"),
                "status_type": str(type(d.get("status"))),
                "scheduled_time": str(d.get("scheduled_time")),
                "scheduled_type": str(type(d.get("scheduled_time"))),
            }
        )

    print(f"Found {len(docs)} reminders for {USER_ID}")
    for item in docs:
        print(item)

    await Database.close_database_connection()


if __name__ == "__main__":
    asyncio.run(main())
