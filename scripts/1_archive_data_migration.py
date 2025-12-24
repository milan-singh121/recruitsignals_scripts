import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime
import motor.motor_asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("migration.log")],
)
log = logging.getLogger(__name__)


async def migrate_collection(
    source_db: motor.motor_asyncio.AsyncIOMotorDatabase,
    archive_db: motor.motor_asyncio.AsyncIOMotorDatabase,
    collection_name: str,
    semaphore: asyncio.Semaphore,
    chunk_size: int,
    source_date_obj: datetime,
):
    """
    Migrates a single collection by streaming from source and
    bulk-inserting into the archive. Appends data, does not drop.
    """
    async with semaphore:
        log.info(
            f"[{collection_name}] Starting migration for {source_date_obj.strftime('%Y-%m')}..."
        )
        try:
            source_col = source_db[collection_name]
            archive_col = archive_db[collection_name]

            cursor = source_col.find({})
            chunk = []
            total_docs = 0

            async for doc in cursor:
                doc["source_date"] = source_date_obj
                chunk.append(doc)

                if len(chunk) >= chunk_size:
                    await archive_col.insert_many(chunk, ordered=False)
                    total_docs += len(chunk)
                    chunk = []

            if chunk:
                await archive_col.insert_many(chunk, ordered=False)
                total_docs += len(chunk)

            log.info(
                f"[{collection_name}] COMPLETED migration. Total docs appended: {total_docs}"
            )
            return total_docs

        except Exception as e:
            log.error(f"[{collection_name}] FAILED migration: {e}")
            return 0


async def main():
    """
    Main function to coordinate the entire migration.
    """
    log.info("Starting MongoDB migration process...")

    parser = argparse.ArgumentParser(
        description="Migrate MongoDB data from one DB to an archive DB, tagging with a date."
    )
    parser.add_argument(
        "--month",
        required=True,
        type=str,
        help="The month of the data being archived, in YYYY-MM format (e.g., '2025-09').",
    )
    args = parser.parse_args()

    try:
        source_date_obj = datetime.strptime(args.month, "%Y-%m")
        log.info(
            f"All migrated data will be tagged with source_date: {source_date_obj}"
        )
    except ValueError:
        log.error(f"Invalid date format: '{args.month}'. Please use YYYY-MM format.")
        return

    load_dotenv()
    SOURCE_MONGO_URI = os.environ.get("MONGO_URI")
    SOURCE_DB_NAME = os.environ.get("MONGO_DB")
    ARCHIVE_MONGO_URI = os.environ.get("MONGO_URI")
    ARCHIVE_DB_NAME = os.environ.get("MONGO_DB_ARCHIVE")

    CHUNK_SIZE = int(os.environ.get("MIGRATION_CHUNK_SIZE", 50000))
    MAX_CONCURRENT_COLLECTIONS = int(os.environ.get("MAX_CONCURRENT_COLLECTIONS", 10))

    if not all([SOURCE_MONGO_URI, SOURCE_DB_NAME, ARCHIVE_MONGO_URI, ARCHIVE_DB_NAME]):
        log.error(
            "Missing environment variables. Ensure all four are set: "
            "SOURCE_MONGO_URI, SOURCE_DB_NAME, ARCHIVE_MONGO_URI, ARCHIVE_DB_NAME"
        )
        return

    source_client = None
    archive_client = None

    try:
        log.info(f"Connecting to SOURCE: {SOURCE_DB_NAME}...")
        source_client = motor.motor_asyncio.AsyncIOMotorClient(SOURCE_MONGO_URI)
        source_db = source_client[SOURCE_DB_NAME]
        await source_client.admin.command("ping")

        log.info(f"Connecting to ARCHIVE: {ARCHIVE_DB_NAME}...")
        archive_client = motor.motor_asyncio.AsyncIOMotorClient(ARCHIVE_MONGO_URI)
        archive_db = archive_client[ARCHIVE_DB_NAME]
        await archive_client.admin.command("ping")

        log.info("Connections successful.")

        all_collections = await source_db.list_collection_names()
        collections_to_migrate = [
            c for c in all_collections if not c.startswith("system.")
        ]

        if not collections_to_migrate:
            log.warning(
                f"No collections found in database '{SOURCE_DB_NAME}'. Exiting."
            )
            return

        log.info(f"Found {len(collections_to_migrate)} collections to migrate.")
        log.info(
            f"Settings: Chunk size={CHUNK_SIZE}, Max parallel collections={MAX_CONCURRENT_COLLECTIONS}"
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_COLLECTIONS)
        tasks = []
        for col_name in collections_to_migrate:
            tasks.append(
                asyncio.create_task(
                    migrate_collection(
                        source_db,
                        archive_db,
                        col_name,
                        semaphore,
                        CHUNK_SIZE,
                        source_date_obj,
                    )
                )
            )

        log.info("--- Starting Migration (see progress bar) ---")
        results = await tqdm_asyncio.gather(*tasks)
        log.info("--- Migration Tasks Finished ---")

        total_migrated_docs = sum(results)
        failed_collections = results.count(0)
        successful_collections = len(results) - failed_collections

        log.info("--- MIGRATION SUMMARY ---")
        log.info(f"Data tagged with month: {args.month}")
        log.info(f"Successful collections: {successful_collections}")
        log.info(f"Failed collections:     {failed_collections}")
        log.info(f"Total documents migrated: {total_migrated_docs}")

    except Exception as e:
        log.error(f"A critical error occurred: {e}")
    finally:
        if source_client:
            source_client.close()
            log.info("Source connection closed.")
        if archive_client:
            archive_client.close()
            log.info("Archive connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Migration interrupted by user.")


# python3 archive_data_migration.py --month 2025-09
# The year-month acts as a key in the documents that are migrated, marking them with the month to which those documents belong.
