import os
import sys
import asyncio
import logging
import json
import argparse
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

import pandas as pd
import motor.motor_asyncio
from pymongo import UpdateOne
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# --- Configuration ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("company_signals.log"),
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


def get_comparison_metrics(series: pd.Series) -> dict:
    """Calculates time-based comparison metrics for a given data series."""
    series = series.dropna().sort_index()
    if len(series) < 2:
        return {}

    metrics = {}
    latest_date = series.index.max()
    latest_value = series.loc[latest_date]

    for months in [1, 3, 6, 12]:
        prev_date = latest_date - relativedelta(months=months)
        available_dates = series.index[series.index <= prev_date]

        if not available_dates.empty:
            actual_prev_date = available_dates.max()
            prev_value = series.loc[actual_prev_date]
            abs_change = latest_value - prev_value
            pct_change = (
                (abs_change / prev_value * 100)
                if prev_value > 0
                else (99999.0 if latest_value > 0 else 0.0)
            )
            metrics[f"{months}_month_comparison"] = {
                "latest_date": latest_date.strftime("%Y-%m"),
                "previous_date": actual_prev_date.strftime("%Y-%m"),
                "latest_value": float(latest_value),
                "previous_value": float(prev_value),
                "absolute_change": float(abs_change),
                "percentage_change": round(pct_change, 2),
            }
    return metrics


def flatten_country_data(monthly_list: list) -> pd.DataFrame:
    """Flattens and pivots the nested country data into a DataFrame."""
    flat_list = []
    for month_data in monthly_list:
        date = month_data.get("date")
        nested_list = month_data.get("employees_count_by_country", [])
        if hasattr(nested_list, "tolist"):
            nested_list = nested_list.tolist()
        if not date or not nested_list:
            continue
        for item in nested_list:
            if item and "country" in item and item.get("country"):
                flat_list.append(
                    {
                        "date": date,
                        "category": item["country"],
                        "value": item["employee_count"],
                    }
                )
    if not flat_list:
        return pd.DataFrame()
    df = pd.DataFrame(flat_list)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    return df.pivot_table(
        index="date", columns="category", values="value", aggfunc="sum"
    )


def process_category_data(df: pd.DataFrame) -> dict:
    """Processes a DataFrame to generate comparison metrics for each column."""
    category_results = {}
    for column in df.columns:
        if metrics := get_comparison_metrics(df[column].astype(float)):
            metrics_by_period = {}
            for period, data in metrics.items():
                metrics_by_period[period] = {"metrics": data}
            category_results[column] = metrics_by_period
    return category_results


def generate_comprehensive_analysis(company_data: dict) -> dict:
    """Generates a structured report of data signals. (Synchronous, CPU-bound)"""
    analysis_report = {
        "company_id": company_data.get("company_id"),
        "company_name": company_data.get("company_name"),
        "analysis_date": datetime.now().isoformat(),
        "signals": {},
    }

    overall_list = company_data.get("employees_count_inferred_by_month", [])
    if overall_list is not None and len(overall_list) > 0:
        if hasattr(overall_list, "tolist"):
            overall_list = overall_list.tolist()
        df_overall = pd.DataFrame(overall_list)
        if not df_overall.empty and "date" in df_overall.columns:
            df_overall["date"] = pd.to_datetime(df_overall["date"], format="%Y%m")
            df_overall = df_overall.set_index("date").rename(
                columns={"employees_count_inferred": "Overall Headcount"}
            )
            analysis_report["signals"]["overall_headcount"] = process_category_data(
                df_overall
            )

    dict_based_categories = {
        "by_seniority": "employees_count_breakdown_by_seniority_by_month",
        "by_department": "employees_count_breakdown_by_department_by_month",
        "by_region": "employees_count_breakdown_by_region_by_month",
    }
    for report_key, data_key in dict_based_categories.items():
        monthly_list = company_data.get(data_key, [])
        if monthly_list is not None and len(monthly_list) > 0:
            if hasattr(monthly_list, "tolist"):
                monthly_list = monthly_list.tolist()
            nested_key = data_key.replace("_by_month", "")
            flat_list = [
                {"date": item["date"], **item[nested_key]}
                for item in monthly_list
                if item and nested_key in item and "date" in item
            ]
            if flat_list:
                df = pd.DataFrame(flat_list).set_index("date")
                df.index = pd.to_datetime(df.index, format="%Y%m")
                df.columns = [c.replace("employees_count_", "") for c in df.columns]
                df = df.loc[:, (df != 0).any(axis=0)]
                if not df.empty:
                    analysis_report["signals"][report_key] = process_category_data(df)

    country_list = company_data.get("employees_count_by_country_by_month", [])
    if country_list is not None and len(country_list) > 0:
        if hasattr(country_list, "tolist"):
            country_list = country_list.tolist()
        df_country = flatten_country_data(country_list)
        if not df_country.empty:
            analysis_report["signals"]["by_country"] = process_category_data(df_country)

    return analysis_report


def restructure_signals_for_mongo(signals_analysis: dict) -> dict:
    """
    Converts the nested analysis report into the nested dict structure
    for MongoDB, as seen in your `restructure_insights` function.
    """
    signals_dict = defaultdict(lambda: defaultdict(dict))

    for category_type, sub_categories in signals_analysis.get("signals", {}).items():
        for category_name, periods in sub_categories.items():
            for period_name, data in periods.items():
                metrics = data.get("metrics", {})
                if metrics:
                    cperiod = period_name.replace("_comparison", "")
                    signals_dict[category_type][category_name][cperiod] = metrics

    return dict(signals_dict)


class MongoSignalETL:
    """
    Handles the full ETL pipeline from a source collection
    to a target 'employee_trends' collection.
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
        log.info("Indexes ensured.")

    def close(self):
        if self.client:
            self.client.close()
            log.info("MongoDB connection closed.")

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

    async def transform_and_prepare_upsert(
        self, doc: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> Optional[UpdateOne]:
        """
        This is the core ETL for a *single* company.
        """
        company_id = doc.get("company_id")
        if not company_id:
            log.warning(f"Skipping doc, missing company_id: {doc.get('_id')}")
            return None

        async with semaphore:
            try:
                analysis_report = await asyncio.to_thread(
                    generate_comprehensive_analysis, doc
                )

                nested_signals = restructure_signals_for_mongo(analysis_report)

                final_doc = {
                    "company_id": doc.get("company_id"),
                    "company_name": doc.get("company_name"),
                    "message_date": datetime.utcnow().isoformat(),
                    "website": doc.get("website"),
                    "industry": doc.get("industry"),
                    "type": doc.get("type"),
                    "hq_country": doc.get("hq_country"),
                    "linkedin_url": doc.get("linkedin_url"),
                    "description_enriched": doc.get("description_enriched"),
                    "company_size": clean_size_range(doc.get("size_range")),
                    "signals": nested_signals,
                }

                query = {"company_id": company_id}
                update = {"$set": final_doc}
                return UpdateOne(query, update, upsert=True)

            except Exception as e:
                log.error(f"Failed to process company {company_id}: {e}", exc_info=True)
                return None

    async def run_pipeline(self, batch_size: int, workers: int):
        """
        Main method to run the entire ETL pipeline.
        Fetches ALL companies from source and upserts them into target.
        """
        try:
            await self.connect()

            # Concurrency limiter for the analysis tasks
            semaphore = asyncio.Semaphore(workers)

            log.info(
                f"Starting full refresh/upsert from '{self.source_collection_name}'..."
            )

            cursor = self.source_coll.find({})

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
                f"✅ All companies processed. Pipeline complete. Total upserted: {total_processed}"
            )

        except Exception as e:
            log.critical(f"💥 Pipeline failed: {e}", exc_info=True)
        finally:
            self.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Run the company signals ETL pipeline."
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Docs to process per batch."
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel analysis tasks."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB")
    SOURCE_COLLECTION = os.getenv("COMPANY_COLLECTION")
    TARGET_COLLECTION = os.getenv("EMPLOYEE_TRENDS_COLLECTION")

    if not MONGO_URI or not MONGO_DB:
        log.critical("MONGO_URI and MONGO_DB must be set in your .env file. Exiting.")
        return

    log.info(
        f"🚀 Starting ETL pipeline: {SOURCE_COLLECTION} -> {TARGET_COLLECTION}. "
        f"Workers={args.workers}, BatchSize={args.batch_size}"
    )

    pipeline = MongoSignalETL(MONGO_URI, MONGO_DB, SOURCE_COLLECTION, TARGET_COLLECTION)
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


# python3 employee_trends.py --batch-size 1000 --workers 20
