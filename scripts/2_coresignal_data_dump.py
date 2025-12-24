import os
import sys
import asyncio
import logging
import json
import argparse
from datetime import datetime
import motor.motor_asyncio
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from pymongo import ReplaceOne


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("upsert.log")],
)
log = logging.getLogger(__name__)


async def process_file_upsert(
    filepath: Path,
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    collection_name: str,
    key_field: str,
    semaphore: asyncio.Semaphore,
    batch_size: int,
    source_date_obj: datetime,  # <-- NEW
) -> int:
    """
    Worker function to stream a single file and perform bulk upserts.
    Adds a 'source_date' field to every document.
    """
    async with semaphore:
        log.info(f"[{collection_name}] Processing file: {filepath.name}")
        collection = db[collection_name]
        ops_chunk = []
        total_docs_in_file = 0

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning(
                            f"[{collection_name}] Skipping bad JSON line in {filepath.name}"
                        )
                        continue

                    key_value = doc.get(key_field)
                    if not key_value:
                        log.warning(
                            f"[{collection_name}] Skipping doc with missing key '{key_field}' in {filepath.name}"
                        )
                        continue

                    # 🆕 Add source_date field
                    doc["source_date"] = source_date_obj

                    op = ReplaceOne(
                        filter={key_field: key_value}, replacement=doc, upsert=True
                    )
                    ops_chunk.append(op)

                    if len(ops_chunk) >= batch_size:
                        await collection.bulk_write(ops_chunk, ordered=False)
                        total_docs_in_file += len(ops_chunk)
                        ops_chunk = []

                if ops_chunk:
                    await collection.bulk_write(ops_chunk, ordered=False)
                    total_docs_in_file += len(ops_chunk)

            log.info(
                f"[{collection_name}] Finished file: {filepath.name}. Upserted {total_docs_in_file} docs."
            )
            return total_docs_in_file

        except Exception as e:
            log.error(f"[{collection_name}] FAILED processing {filepath.name}: {e}")
            return 0


async def run_directory_upsert(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    directory_path: str,
    collection_name: str,
    key_field: str,
    semaphore: asyncio.Semaphore,
    batch_size: int,
    source_date_obj: datetime,
):
    """
    Finds all .jsonl files in a directory and coordinates their parallel processing.
    """
    log.info(f"--- Starting upsert for '{collection_name}' from {directory_path} ---")
    path = Path(directory_path)
    files = list(path.glob("*.jsonl")) + list(path.glob("*.json"))

    if not files:
        log.warning(f"No .json or .jsonl files found in {directory_path}. Skipping.")
        return

    log.info(f"Found {len(files)} files to process for '{collection_name}'.")

    tasks = []
    for f in files:
        if f.is_file():
            tasks.append(
                process_file_upsert(
                    f,
                    db,
                    collection_name,
                    key_field,
                    semaphore,
                    batch_size,
                    source_date_obj,
                )
            )

    results = await tqdm_asyncio.gather(*tasks, desc=f"Upserting to {collection_name}")

    total_docs_processed = sum(results)
    log.info(f"--- '{collection_name}' Summary ---")
    log.info(
        f"Processed {len(files)} files. Total documents upserted: {total_docs_processed}"
    )
    log.info(f"--- Finished '{collection_name}' ---")


async def main():
    """
    Main function to coordinate the entire upsert process.
    """
    parser = argparse.ArgumentParser(
        description="Production-level script to upsert JSONL data into MongoDB."
    )
    parser.add_argument(
        "--people-dir", type=str, help="Directory containing 'people' JSONL files."
    )
    parser.add_argument(
        "--company-dir", type=str, help="Directory containing 'company' JSONL files."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents to upsert in one batch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max number of files to process at the same time.",
    )

    parser.add_argument(
        "--month",
        required=True,
        type=str,
        help="Month of the dataset being upserted, in YYYY-MM format (e.g., '2025-10').",
    )
    args = parser.parse_args()

    if not args.people_dir and not args.company_dir:
        log.error("You must provide --people-dir, --company-dir, or both. Exiting.")
        parser.print_help()
        return

    try:
        source_date_obj = datetime.strptime(args.month, "%Y-%m")
        log.info(
            f"All upserted documents will be tagged with source_date: {source_date_obj}"
        )
    except ValueError:
        log.error(f"Invalid month format '{args.month}'. Please use YYYY-MM.")
        return

    load_dotenv()
    MONGO_URI = os.environ.get("MONGO_URI")
    MONGO_DB = os.environ.get("MONGO_DB")

    if not MONGO_URI or not MONGO_DB:
        log.error("MONGO_URI and MONGO_DB must be set in your .env file.")
        return

    client = None
    try:
        log.info(f"Connecting to MongoDB at {MONGO_URI.split('@')[-1]}...")
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        db = client[MONGO_DB]
        await client.admin.command("ping")
        log.info(f"Successfully connected to database '{MONGO_DB}'.")

        semaphore = asyncio.Semaphore(args.concurrency)

        if args.company_dir:
            await run_directory_upsert(
                db,
                args.company_dir,
                collection_name="company",
                key_field="company_id",
                semaphore=semaphore,
                batch_size=args.batch_size,
                source_date_obj=source_date_obj,
            )

        if args.people_dir:
            await run_directory_upsert(
                db,
                args.people_dir,
                collection_name="people",
                key_field="member_id",
                semaphore=semaphore,
                batch_size=args.batch_size,
                source_date_obj=source_date_obj,
            )

        log.info("--- Upsert process completed successfully. ---")

    except Exception as e:
        log.error(f"A critical error occurred: {e}")
    finally:
        if client:
            client.close()
            log.info("MongoDB connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Script interrupted by user")


# python3 2_coresignal_data_dump.py --company-dir "/Users/trentino/Work/RecruitSignalsData/data/coresignal_data/October 2025/company/" --month 2025-10

# To upsert only people data:
# python3 2_coresignal_data_dump.py --people-dir "/Users/trentino/Work/RecruitSignalsData/data/coresignal_data/October 2025/clean_members_UK_NE_jsonl/" --month 2025-10


# python3 2_coresignal_data_dump.py \
#   --people-dir "/Users/trentino/Work/RecruitSignalsData/data/coresignal_data/October 2025/people/" \
#   --company-dir "/Users/trentino/Work/RecruitSignalsData/data/coresignal_data/October 2025/company/" \
#   --month 2025-10
