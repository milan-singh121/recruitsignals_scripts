import os
import sys
import logging
import argparse
import time
import json
import re
from typing import Optional, List, Dict, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, BulkWriteError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")
COLLECTION_NAME = os.getenv("PEOPLE_COLLECTION")

CXO_FLAG_KEY = "is_cxo"

CXO_TITLE_KEYWORDS = [
    # English
    r"\bceo\b",
    r"\bcfo\b",
    r"\bcto\b",
    r"\bcoo\b",
    r"\bcio\b",
    r"\bcmo\b",
    r"\bcro\b",
    r"\bchief.*officer\b",
    r"\bfounder\b",
    r"\bco-founder\b",
    r"\bcofounder\b",
    r"\bowner\b",
    r"\bproprietor\b",
    r"\bchairman\b",
    r"\bpresident\b",
    r"\bgeschäftsführer\b",
    # French
    r"\bpdg\b",
    r"directeur général",
    r"directeur financier",
    r"directeur technique",
    r"\bfondateur\b",
    r"\bcofondateur\b",
    r"\bpropriétaire\b",
    r"\bgérant\b",
    r"\bprésident\b",
    # German
    r"geschäftsführer",
    r"vorstandsvorsitzender",
    r"finanzvorstand",
    r"technologievorstand",
    r"\bgründer\b",
    r"\bmitbegründer\b",
    r"\binhaber\b",
    # Dutch
    r"\bbestuurder\b",
    r"algemeen directeur",
    r"financieel directeur",
    r"technisch directeur",
    r"\boprichter\b",
    r"\bmede-oprichter\b",
    r"\beigenaar\b",
    r"directeur-eigenaar",
]
CXO_TITLE_PATTERN = re.compile("|".join(CXO_TITLE_KEYWORDS), re.IGNORECASE)

CXO_MGMT_LEVEL_SET = {
    "founder",
    "c-level",
    "cxo",
    "owner",
    "partner",
    "executive",
    "president",
    "directeur",
    "vorstand",
}

CXO_DEPT_SET = {"c-suite", "executive", "board", "management board", "directie"}


def check_is_cxo(
    title: Optional[str], mgmt_level: Optional[str], department: Optional[str]
) -> bool:
    """
    Checks if a person is a CXO, Founder, or Owner based on keyword logic.
    """
    try:
        if isinstance(mgmt_level, str) and mgmt_level.lower() in CXO_MGMT_LEVEL_SET:
            return True

        if isinstance(department, str) and department.lower() in CXO_DEPT_SET:
            return True

        if isinstance(title, str) and CXO_TITLE_PATTERN.search(title):
            return True

    except Exception as e:
        logging.warning(f"Error during keyword check: {e}. Defaulting to False.")

    return False


class MongoDBHandler:
    """Handles all operations with the MongoDB database."""

    def __init__(self, uri: str, db_name: str, collection_name: str):
        if not uri:
            raise ValueError("MongoDB URI must be provided.")
        try:
            self.client = MongoClient(uri)
            self.client.admin.command("ismaster")
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            logging.info(
                f"✅ Connected to MongoDB. DB: '{db_name}', Collection: '{collection_name}'."
            )
        except ConnectionFailure as e:
            logging.error(f"❌ Could not connect to MongoDB: {e}", exc_info=True)
            raise

    def get_unprocessed_documents(
        self, flag_key: str, limit: int
    ) -> Iterator[Dict[str, Any]]:
        """
        Gets a batch of documents where the cxo_flag_key does not exist.
        """
        query = {flag_key: {"$exists": False}}
        projection = {
            "member_job_title": 1,
            "member_management_level": 1,
            "member_department": 1,
        }
        logging.info(
            f"Fetching up to {limit} unprocessed docs where '{flag_key}' does not exist."
        )
        try:
            return self.collection.find(query, projection).limit(limit)
        except Exception as e:
            logging.error(f"Error fetching documents: {e}")
            return iter([])

    def bulk_update_documents(self, update_operations: List[UpdateOne]) -> None:
        """
        Performs a bulk write operation to update document flags.
        """
        if not update_operations:
            logging.info("No update operations to perform.")
            return
        try:
            result = self.collection.bulk_write(update_operations)
            logging.info(
                f"📈 Bulk update successful: {result.modified_count} documents updated."
            )
        except BulkWriteError as bwe:
            logging.error(f"❌ A bulk write error occurred: {bwe.details}")
        except Exception:
            logging.exception("❌ An unexpected error during bulk update.")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("flag_cxo_profiles.log"),
            logging.StreamHandler(),
        ],
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process LinkedIn profiles to flag CXOs using keyword logic."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "production"],
        required=True,
        help="Run mode. 'test' processes --limit docs. 'production' runs until complete.",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Total docs to process in 'test' mode."
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Docs to process per batch."
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_arguments()
    logging.info(
        f"🚀 Starting script in '{args.mode}' mode. Batch size: {args.batch_size}."
    )

    try:
        if not MONGO_URI:
            logging.error("MONGO_CLIENT_URL not found in .env file.")
            sys.exit(1)

        mongo_handler = MongoDBHandler(MONGO_URI, DB_NAME, COLLECTION_NAME)

        process_limit = args.limit if args.mode == "test" else float("inf")
        total_processed = 0

        while total_processed < process_limit:
            documents = list(
                mongo_handler.get_unprocessed_documents(CXO_FLAG_KEY, args.batch_size)
            )
            if not documents:
                logging.info("✅ No more documents to process. Exiting.")
                break

            logging.info(f"Processing batch of {len(documents)} documents...")
            update_operations = []

            for doc in documents:
                try:
                    doc_id = doc["_id"]
                    title = doc.get("member_job_title")
                    management_level = doc.get("member_management_level")
                    member_department = doc.get("member_department")

                    is_cxo_status = check_is_cxo(
                        title, management_level, member_department
                    )

                    update_operations.append(
                        UpdateOne(
                            {"_id": doc_id}, {"$set": {CXO_FLAG_KEY: is_cxo_status}}
                        )
                    )
                except Exception as e:
                    logging.error(
                        f"Error processing document {doc.get('_id')}: {e}",
                        exc_info=True,
                    )

            if update_operations:
                mongo_handler.bulk_update_documents(update_operations)
                total_processed += len(update_operations)

            if args.mode == "test" and total_processed >= args.limit:
                logging.info(f"Reached test limit of {args.limit} documents.")
                break

        logging.info(
            f"🎉 Script finished. Total documents processed: {total_processed}"
        )

    except Exception as e:
        logging.critical(
            f"💥 A critical error occurred in the main process: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()

# python3 process_profiles_logic.py --mode test --limit 100 --batch-size 50
# python3 process_profiles_logic.py --mode production --batch-size 500
