import os
import sys
import asyncio
import logging
import json
import argparse
import time
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd
import motor.motor_asyncio
from pymongo import UpdateOne
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("funding_etl.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def clean_size_range(value: Optional[str]) -> Optional[str]:
    """Cleans the company size_range string for database storage."""
    if pd.isna(value) or value is None:
        return None
    if value == "Myself Only":
        return "1"
    value = str(value).replace(" employees", "")
    value = str(value).replace(",", "")
    return value


def clean_data_for_mongo(data: Any) -> Any:
    """
    Recursively cleans a data structure to make it BSON-compatible.
    - Replaces float('nan') with None, which becomes 'null' in MongoDB.
    """
    if isinstance(data, dict):
        return {k: clean_data_for_mongo(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_data_for_mongo(item) for item in data]
    if isinstance(data, float) and math.isnan(data):
        return None
    return data


def find_latest_funding_round_sync(funding_rounds: List[Dict]) -> Optional[Dict]:
    """
    Finds the latest funding round from a list based on 'announced_date'.
    This is a synchronous, CPU-bound helper.
    """
    if not funding_rounds:
        return None

    def parse_date(date_str: Optional[str]) -> datetime:
        try:
            return datetime.strptime(str(date_str), "%Y-%m-%d")
        except (ValueError, TypeError):
            return datetime.min

    try:
        latest_round = max(
            funding_rounds, key=lambda fr: parse_date(fr.get("announced_date"))
        )
        return latest_round
    except Exception as e:
        log.error(f"Error finding latest funding round: {e}")
        return None


def transform_company_data_sync(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The core synchronous transformation logic for a single company document.
    This will be run in a separate thread.
    """
    # 1. Extract base data
    company_id = doc.get("company_id")
    if not company_id:
        return None

    funding_rounds = doc.get("funding_rounds", [])
    if not isinstance(funding_rounds, list) and funding_rounds is not None:
        try:
            funding_rounds = funding_rounds.tolist()
        except:
            log.warning(f"Could not convert funding_rounds to list for {company_id}")
            return None

    if not funding_rounds:
        return None

    latest_round = find_latest_funding_round_sync(funding_rounds)
    if not latest_round:
        return None

    signal_doc = {
        "company_id": company_id,
        "company_name": doc.get("company_name"),
        "website": doc.get("website"),
        "linkedin_url": doc.get("linkedin_url"),
        "industry": doc.get("industry"),
        "hq_country": doc.get("hq_country"),
        "company_description": doc.get("description_enriched"),
        "company_size": clean_size_range(doc.get("size_range")),
        "message_date": datetime.utcnow().isoformat(),
        "funding_round_name": latest_round.get("name"),
        "funding_round_date": latest_round.get("announced_date"),
        "funding_round_amount": latest_round.get("amount_raised"),
        "funding_round_currency": latest_round.get("amount_raised_currency"),
        "funding_round_num_investors": latest_round.get("num_investors"),
        "funding_round_lead_investors": latest_round.get("lead_investors", []),
    }

    clean_funding_stage = (
        lambda x: x.split(" - ", 1)[0] if isinstance(x, str) and " - " in x else x
    )(signal_doc.get("funding_round_name"))

    # Strip whitespace safely
    signal_doc["funding_stage"] = (
        clean_funding_stage.strip()
        if isinstance(clean_funding_stage, str)
        else clean_funding_stage
    )
    return clean_data_for_mongo(signal_doc)


class MongoFundingETL:
    """
    Handles the full ETL pipeline from a source collection
    to a target 'funding_trends' collection.
    """

    def __init__(self, uri: str, db_name: str, source_col: str, target_col: str):
        if not uri:
            raise ValueError("MongoDB URI must be provided.")
        self.uri = uri
        self.db_name = db_name
        self.source_collection_name = source_col
        self.target_collection_name = target_col
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
        self.source_coll: Optional[motor.motor_asyncio.AsyncIOMotorCollection] = None
        self.target_coll: Optional[motor.motor_asyncio.AsyncIOMotorCollection] = None

    async def connect(self):
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.uri)
            await self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.source_coll = self.db[self.source_collection_name]
            self.target_coll = self.db[self.target_collection_name]
            log.info(
                f"✅ Connected to MongoDB. DB: '{self.db_name}'. "
                f"Source: '{self.source_collection_name}', Target: '{self.target_collection_name}'"
            )
            await self.create_indexes()
        except Exception as e:
            log.error(f"❌ Could not connect to MongoDB: {e}", exc_info=True)
            raise

    async def create_indexes(self):
        """Creates the necessary indexes on the target collection."""
        log.info(
            f"Ensuring indexes on target collection '{self.target_collection_name}'..."
        )
        await self.target_coll.create_index([("company_id", 1)], unique=True)
        log.info("Index on 'company_id' ensured.")

    def close(self):
        if self.client:
            self.client.close()
            log.info("MongoDB connection closed.")

    async def transform_and_prepare_upsert(
        self, doc: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> Optional[UpdateOne]:
        """
        This is the core ETL for a *single* company.
        """
        async with semaphore:
            try:
                final_doc = await asyncio.to_thread(transform_company_data_sync, doc)

                if not final_doc:
                    return None

                query = {"company_id": final_doc["company_id"]}
                update = {"$set": final_doc}
                return UpdateOne(query, update, upsert=True)

            except Exception as e:
                log.error(
                    f"Failed to process company {doc.get('company_id')}: {e}",
                    exc_info=True,
                )
                return None

    def process_company_batch(
        self,
        docs_batch: List[Dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> List[asyncio.Task]:
        """
        Creates processing tasks for a batch of company documents.
        """
        tasks = [
            asyncio.create_task(self.transform_and_prepare_upsert(doc, semaphore))
            for doc in docs_batch
        ]
        return tasks

    async def run_pipeline(self, batch_size: int, workers: int):
        """
        Main method to run the entire ETL pipeline.
        Fetches ALL companies from source, filters, and upserts them into target.
        """
        try:
            await self.connect()

            semaphore = asyncio.Semaphore(workers)

            log.info(
                f"Starting full refresh/upsert from '{self.source_collection_name}'..."
            )

            query = {
                "funding_rounds": {"$exists": True, "$ne": [], "$not": {"$size": 0}}
            }
            cursor = self.source_coll.find(query)

            total_processed = 0
            docs_batch = []

            async for doc in cursor:
                docs_batch.append(doc)

                if len(docs_batch) >= batch_size:
                    log.info(f"Processing batch of {len(docs_batch)} companies...")

                    tasks = self.process_company_batch(docs_batch, semaphore)
                    results = await tqdm_asyncio.gather(*tasks, desc="Analyzing Batch")
                    operations = [op for op in results if op is not None]

                    if operations:
                        log.info(
                            f"Performing bulk upsert of {len(operations)} documents..."
                        )
                        try:
                            result = await self.target_coll.bulk_write(
                                operations, ordered=False
                            )
                            log.info(
                                f"✅ Bulk upsert complete: "
                                f"Inserted={result.upserted_count}, Updated={result.modified_count}"
                            )
                            total_processed += len(operations)
                        except Exception as e:
                            log.error(f"Bulk write error: {e}", exc_info=True)

                    docs_batch = []

            if docs_batch:
                log.info(f"Processing final batch of {len(docs_batch)} companies...")
                tasks = self.process_company_batch(docs_batch, semaphore)
                results = await tqdm_asyncio.gather(
                    *tasks, desc="Analyzing Final Batch"
                )
                operations = [op for op in results if op is not None]

                if operations:
                    log.info(
                        f"Performing bulk upsert of {len(operations)} documents..."
                    )
                    try:
                        result = await self.target_coll.bulk_write(
                            operations, ordered=False
                        )
                        log.info(
                            f"✅ Bulk upsert complete: "
                            f"Inserted={result.upserted_count}, Updated={result.modified_count}"
                        )
                        total_processed += len(operations)
                    except Exception as e:
                        log.error(f"Bulk write error: {e}", exc_info=True)

            log.info(
                f"✅ Funding ETL complete. Total companies with funding processed: {total_processed}"
            )

        except Exception as e:
            log.critical(f"💥 Pipeline failed: {e}", exc_info=True)
        finally:
            self.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Run the company funding trends ETL pipeline."
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Docs to process per batch."
    )
    parser.add_argument(
        "--workers", type=int, default=20, help="Number of parallel analysis tasks."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB")
    SOURCE_COLLECTION = os.getenv("COMPANY_COLLECTION")
    TARGET_COLLECTION = os.getenv("FUNDING_TRENDS_COLLECTION")

    if not MONGO_URI or not MONGO_DB:
        log.critical("MONGO_URI and MONGO_DB must be set in your .env file. Exiting.")
        return

    log.info(
        f"🚀 Starting ETL pipeline: {SOURCE_COLLECTION} -> {TARGET_COLLECTION}. "
        f"Workers={args.workers}, BatchSize={args.batch_size}"
    )

    pipeline = MongoFundingETL(
        MONGO_URI, MONGO_DB, SOURCE_COLLECTION, TARGET_COLLECTION
    )
    await pipeline.run_pipeline(args.batch_size, args.workers)


if __name__ == "__main__":
    if not os.getenv("MONGO_URI"):
        log.info("Loading .env file...")
        if not load_dotenv():
            log.error("Could not find .env file.")
        else:
            log.info(".env file loaded.")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Script interrupted by user.")


# python3 funding_trends.py --batch-size 1000 --workers 20
