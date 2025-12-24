import os
import sys
import asyncio
import logging
import math
from typing import List, Dict, Optional
from collections import defaultdict

import pandas as pd
import motor.motor_asyncio
from pymongo import UpdateOne, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure, BulkWriteError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

LOG_FILE = os.getenv("LOG_FILE", "signals_overview.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


class MongoLoader:
    """
    Handles all async MongoDB connections and operations.
    """

    def __init__(self, uri: str, db_name: str):
        if not uri or not db_name:
            log.error("MongoDB URI and DB Name must be provided.")
            raise ValueError("MongoDB URI and DB Name must be provided.")
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
        self.uri = uri
        self.db_name = db_name
        self.lock = asyncio.Lock()

    async def connect(self):
        """Establishes the asynchronous connection to the MongoDB server."""

        if self.db is not None:
            return

        async with self.lock:
            if self.db is not None:
                return

            try:
                log.info(f"Connecting to MongoDB. DB: '{self.db_name}'.")
                self.client = motor.motor_asyncio.AsyncIOMotorClient(
                    self.uri, serverSelectionTimeoutMS=5000
                )
                await self.client.admin.command("ping")
                self.db = self.client[self.db_name]
                log.info(f"✅ Connected to MongoDB. DB: '{self.db_name}'.")
            except ConnectionFailure as e:
                log.error(f"❌ Could not connect to MongoDB: {e}", exc_info=True)
                self.client = None
                self.db = None
                raise
            except Exception as e:
                log.error(
                    f"❌ An unexpected error occurred during connection: {e}",
                    exc_info=True,
                )
                self.client = None
                self.db = None
                raise

    async def close(self):
        """Closes the connection to the MongoDB server."""
        if self.client:
            self.client.close()
            log.info("MongoDB connection closed.")

    async def load_collection_to_df(
        self, collection_name: str, query: Dict = None, projection: Dict = None
    ) -> pd.DataFrame:
        """Loads an entire collection into a pandas DataFrame."""
        if self.db is None:
            await self.connect()

        if not collection_name:
            log.error("Collection name must be provided.")
            return pd.DataFrame()

        collection = self.db[collection_name]
        log.info(f"Loading collection '{collection_name}' into DataFrame...")

        cursor = collection.find(query or {}, projection)
        data = await cursor.to_list(length=None)

        log.info(f"Loaded {len(data)} documents from '{collection_name}'.")
        return pd.DataFrame(data)

    async def load_all_sources(
        self, employee_col: str, funding_col: str, leadership_col: str
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """Loads all three source collections in parallel."""
        task_employee = self.load_collection_to_df(
            employee_col,
            projection={
                "company_id": 1,
                "company_name": 1,
                "company_size": 1,
                "hq_country": 1,
                "industry": 1,
                "signals": 1,
            },
        )
        task_funding = self.load_collection_to_df(
            funding_col,
            projection={
                "company_id": 1,
                "company_name": 1,
                "company_size": 1,
                "funding_round_amount": 1,
                "funding_round_currency": 1,
                "funding_round_date": 1,
                "funding_round_lead_investors": 1,
                "funding_round_name": 1,
                "funding_stage": 1,
            },
        )
        # --- NEW ---
        # Load the leadership change data. We only need the company_id.
        task_leadership = self.load_collection_to_df(
            leadership_col,
            projection={
                "company_id": 1,
                "_id": 0,  # Exclude _id
            },
        )

        df_employee, df_funding, df_leadership = await asyncio.gather(
            task_employee, task_funding, task_leadership
        )
        return df_employee, df_funding, df_leadership

    async def dump_to_mongodb(self, data_list: List[Dict], collection_name: str):
        """Dumps the aggregated data into a MongoDB collection using async bulk upserts."""
        if self.db is None:
            await self.connect()

        if not data_list:
            log.warning("No data provided to dump to MongoDB. Skipping.")
            return

        log.info(
            f"Attempting to dump {len(data_list)} aggregated reports to '{collection_name}'..."
        )
        collection = self.db[collection_name]

        index_keys = [(key, ASCENDING) for key in PRIMARY_GROUP_KEYS]
        try:
            await collection.create_index(index_keys, unique=True)
            log.info(
                f"Ensured unique index exists for: {', '.join(PRIMARY_GROUP_KEYS)}"
            )
        except OperationFailure as e:
            log.error(
                f"Failed to create index (this might be ok if it already exists): {e}"
            )

        operations = []
        for doc in data_list:
            query_filter = {key: doc[key] for key in PRIMARY_GROUP_KEYS}
            update_operation = UpdateOne(query_filter, {"$set": doc}, upsert=True)
            operations.append(update_operation)

        log.info(f"Performing bulk upsert of {len(operations)} documents...")
        try:
            result = await collection.bulk_write(operations, ordered=False)
            log.info(
                f"MongoDB bulk write complete. "
                f"Inserted: {result.upserted_count}, "
                f"Updated: {result.modified_count}"
            )
        except BulkWriteError as bwe:
            log.error(f"A bulk write error occurred: {bwe.details}", exc_info=True)
        except Exception as e:
            log.error(
                f"An unexpected error occurred during MongoDB operation: {e}",
                exc_info=True,
            )


def transform_and_merge_data(
    df_employee: pd.DataFrame, df_funding: pd.DataFrame, df_leadership: pd.DataFrame
) -> pd.DataFrame:
    """
    Applies all pandas transformations and merges the dataframes.
    """
    if df_employee.empty:
        log.warning("Employee data is empty. Cannot perform primary merge.")
        return pd.DataFrame()

    merge_keys = [
        "company_id",
        "company_name",
        "company_size",
    ]

    # 1. Merge Employee and Funding (as before)
    df_merged = pd.merge(df_employee, df_funding, on=merge_keys, how="left")

    if FUNDING_STAGE_KEY not in df_merged.columns:
        df_merged[FUNDING_STAGE_KEY] = None

    if not df_leadership.empty:
        df_lead_flags = df_leadership[["company_id"]].copy()
        df_lead_flags["had_leadership_change"] = True

        df_lead_flags = df_lead_flags.drop_duplicates(subset=["company_id"])

        df_merged = pd.merge(df_merged, df_lead_flags, on="company_id", how="left")

        df_merged["had_leadership_change"] = df_merged["had_leadership_change"].fillna(
            False
        )
    else:
        log.warning("Leadership change data is empty. All change reports will be 0.")
        df_merged["had_leadership_change"] = False

    return df_merged


PRIMARY_GROUP_KEYS = ["industry", "hq_country", "company_size"]
SIGNAL_CATEGORIES = ["overall_headcount", "by_seniority", "by_department"]
FUNDING_STAGE_KEY = "funding_stage"


def _clean_value(value, default="Unknown"):
    """Cleans a value to be used as a dictionary key."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, str) and value.lower() == "nan":
        return default
    return str(value)


def aggregate_company_signals(company_data_list: List[Dict]) -> List[Dict]:
    """
    Aggregates company data based on primary keys and analyzes signal and funding data
    at a sub-category level.
    """
    if not isinstance(company_data_list, list):
        log.error("Input data must be a list of dictionaries.")
        return []

    log.info(f"Aggregating {len(company_data_list)} company records...")
    aggregated_data = defaultdict(
        lambda: {
            "total_unique_company_ids": set(),
            "companies_by_signal_subcategory": defaultdict(set),
            "companies_by_funding_stage": defaultdict(set),
            "companies_with_leadership_change": set(),
        }
    )

    for company in tqdm(company_data_list, desc="Aggregating Companies"):
        if not isinstance(company, dict):
            log.warning("Skipping non-dictionary item in input list.")
            continue

        company_id = company.get("company_id")
        if not company_id:
            log.warning(
                f"Skipping company with missing 'company_id'. Data: {company.get('company_name', 'N/A')}"
            )
            continue

        group_key_values = [
            _clean_value(company.get(key)) for key in PRIMARY_GROUP_KEYS
        ]
        group_key = tuple(group_key_values)
        group_data = aggregated_data[group_key]

        group_data["total_unique_company_ids"].add(company_id)

        funding_stage = _clean_value(company.get(FUNDING_STAGE_KEY))
        group_data["companies_by_funding_stage"][funding_stage].add(company_id)

        if company.get("had_leadership_change") is True:
            group_data["companies_with_leadership_change"].add(company_id)

        signals = company.get("signals", {})
        if not isinstance(signals, dict):
            log.warning(
                f"Company ID {company_id} has invalid 'signals' data (not a dict). Skipping signals."
            )
            continue

        found_subcategories = set()

        for cat_type in SIGNAL_CATEGORIES:
            category_data = signals.get(cat_type)
            if not isinstance(category_data, dict):
                continue

            for cat_name in category_data.keys():
                clean_cat_name = _clean_value(cat_name)
                found_subcategories.add((cat_type, clean_cat_name))

        for cat_type, cat_name in found_subcategories:
            group_data["companies_by_signal_subcategory"][(cat_type, cat_name)].add(
                company_id
            )

    log.info(
        f"Successfully processed {len(company_data_list)} records. Generating final report..."
    )

    final_output_list = []
    for group_key, group_data in tqdm(
        aggregated_data.items(), desc="Generating Reports"
    ):
        final_group_report = dict(zip(PRIMARY_GROUP_KEYS, group_key))

        signal_report = defaultdict(dict)
        for (cat_type, cat_name), id_set in group_data[
            "companies_by_signal_subcategory"
        ].items():
            id_list = sorted(list(id_set))
            signal_report[cat_type][cat_name] = {
                "count": len(id_list),
                "company_ids": id_list,
            }

        for cat_type in SIGNAL_CATEGORIES:
            if cat_type not in signal_report:
                signal_report[cat_type] = {}

        clean_signal_report = defaultdict(dict)
        for cat_type, sub_cats in signal_report.items():
            clean_sub_cats = {k: v for k, v in sub_cats.items() if k != "Unknown"}
            clean_signal_report[cat_type] = clean_sub_cats
        final_group_report["signals_by_category_name"] = dict(clean_signal_report)

        funding_report = {}
        for stage, id_set in group_data["companies_by_funding_stage"].items():
            id_list = sorted(list(id_set))
            funding_report[stage] = {"count": len(id_list), "company_ids": id_list}

        clean_funding_report = {
            k: v for k, v in funding_report.items() if k != "Unknown"
        }
        final_group_report["companies_by_funding_stage"] = clean_funding_report

        leadership_change_ids = sorted(
            list(group_data["companies_with_leadership_change"])
        )
        final_group_report["leadership_change_report"] = {
            "count": len(leadership_change_ids),
            "company_ids": leadership_change_ids,
        }

        total_ids_list = sorted(list(group_data["total_unique_company_ids"]))
        final_group_report["total_unique_companies_in_group"] = len(total_ids_list)
        final_group_report["total_unique_company_ids_in_group"] = total_ids_list

        final_output_list.append(final_group_report)

    log.info(
        f"Generated aggregated reports for {len(final_output_list)} unique groups."
    )
    return final_output_list


async def main():
    """Main ETL pipeline."""

    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB")

    EMPLOYEE_COL = os.getenv("EMPLOYEE_TRENDS_COLLECTION")
    FUNDING_COL = os.getenv("FUNDING_TRENDS_COLLECTION")
    TARGET_COL = os.getenv("TARGET_OVERVIEW_COLLECTION")
    LEADERSHIP_COL = os.getenv("LEADERSHIP_CHANGE_COLLECTION")

    if not MONGO_URI or not MONGO_DB:
        log.critical("MONGO_URI and MONGO_DB must be set in your .env file. Exiting.")
        return
    if not TARGET_COL:
        log.critical(
            "TARGET_OVERVIEW_COLLECTION must be set in your .env file. Exiting."
        )
        return
    # --- NEW ---
    if not LEADERSHIP_COL:
        log.critical(
            "MONGO_COL_OUTPUT (for leadership_change) must be set in your .env file. Exiting."
        )
        return

    loader = MongoLoader(MONGO_URI, MONGO_DB)

    try:
        log.info("--- Step 1: Extracting data from MongoDB ---")
        df_employee, df_funding, df_leadership = await loader.load_all_sources(
            EMPLOYEE_COL, FUNDING_COL, LEADERSHIP_COL
        )

        if df_employee.empty:
            log.warning("Source data (employee_trends) is empty. Cannot continue.")
            return

        log.info("--- Step 2: Transforming and Merging data ---")
        combined_df = transform_and_merge_data(df_employee, df_funding, df_leadership)

        if combined_df.empty:
            log.warning("No data found after merging. Exiting.")
            return

        log.info(f"Merge complete. Total combined records: {len(combined_df)}")

        log.info("Filtering out records with unknown company size...")

        filtered_df = combined_df.copy()
        filtered_df["company_size_clean"] = filtered_df["company_size"].apply(
            lambda x: _clean_value(x)
        )

        filtered_df = filtered_df[filtered_df["company_size_clean"] != "Unknown"]

        filtered_df = filtered_df.drop(columns=["company_size_clean"])

        log.info(f"Records remaining after company_size filter: {len(filtered_df)}")

        if filtered_df.empty:
            log.warning(
                "No data remaining after filtering. No aggregates will be generated."
            )
            return

        dict_data = filtered_df.to_dict(orient="records")
        log.info(f"Total records to be aggregated after filtering: {len(dict_data)}")

        log.info("--- Step 3: Aggregating signals ---")
        analysis_result = aggregate_company_signals(dict_data)

        if not analysis_result:
            log.warning("Aggregation produced no results. Exiting.")
            return

        log.info("--- Step 4: Loading data to MongoDB ---")
        await loader.dump_to_mongodb(analysis_result, TARGET_COL)

        log.info("🎉 Signals overview generation complete!")

    except Exception as e:
        log.critical(
            f"💥 A critical error occurred in the main pipeline: {e}", exc_info=True
        )
    finally:
        await loader.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Script interrupted by user.")


# python3 signals.py
