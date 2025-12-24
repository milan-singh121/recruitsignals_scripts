import pymongo
import pandas as pd
import re
import datetime
from typing import Optional
from collections import defaultdict
import json
import sys
import numpy as np
import logging
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv


load_dotenv()
log = logging.getLogger()


required_vars = [
    "MONGO_URI",
    "MONGO_DB_ARCHIVE",
    "MONGO_DB",
    "PEOPLE_COLLECTION",
    "COMPANY_COLLECTION",
    "LEADERSHIP_CHANGE_COLLECTION",
]
if not all(os.getenv(var) for var in required_vars):
    log.error("Missing one or more environment variables in .env file.")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("leadership_change.log"),
    ],
)

LEADERSHIP_KEYWORDS = [
    # English
    r"\bmanager\b",
    r"\bdirector\b",
    r"\bhead of\b",
    r"\bvp\b",
    r"\bvice president\b",
    r"\bchief\b",
    r"\bceo\b",
    r"\bcfo\b",
    r"\bcto\b",
    r"\bcoo\b",
    r"\blead\b",
    r"\bpartner\b",
    r"\bexecutive\b",
    r"\bfounder\b",
    r"\bhead\b",
    r"\bcto\b",
    r"\bchief technology officer\b",
    r"\bchief financial officer\b",
    r"\bleader\b",
    r"\bpresident\b",
    r"\bchief executive officer\b",
    r"\bchief operating officer\b",
    r"\bowner\b",
    r"\bteam leader\b",
    r"\bchairman\b",
    r"\bcofounder\b",
    # French
    r"\bdirecteur\b",
    r"\bdirectrice\b",
    r"\bgérant\b",
    r"\bresponsable\b",
    r"\bchef de\b",
    r"\bpdg\b",
    r"\bdirecteur général\b",
    r"\bdirecteur financier\b",
    r"\bdaf\b",
    r"\bdirecteur technique\b",
    r"\bdirecteur des opérations\b",
    r"\bassocié\b",
    r"\bassociée\b",
    r"\bcadre\b",
    r"\bdirigeant\b",
    r"\bfondateur\b",
    r"\bfondatrice\b",
    r"\bchef\b",
    r"\bprésident\b",
    r"\bprésidente\b",
    r"\bpropriétaire\b",
    r"\bchef d'équipe\b",
    r"\bprésident du conseil\b",
    r"\bcofounder\b",
    r"\bcofondatrice\b",
    # Dutch
    r"\bbestuurder\b",
    r"\bhoofd\b",
    r"\bhoofd van\b",
    r"\bvicepresident\b",
    r"\balgemeen directeur\b",
    r"\bfinancieel directeur\b",
    r"\btechnisch directeur\b",
    r"\boperationeel directeur\b",
    r"\bleider\b",
    r"\bteamleider\b",
    r"\bpartner\b",
    r"\bvennoot\b",
    r"\bleidinggevende\b",
    r"\bbestuurslid\b",
    r"\boprichter\b",
    r"\beigenaar\b",
    r"\bploegleider\b",
    r"\bvoorzitter\b",
    r"\bmede-oprichter\b",
    # German
    r"\bLeiter\b",
    r"\bLeiterin\b",
    r"\bDirektor\b",
    r"\bDirektorin\b",
    r"\bGeschäftsführer\b",
    r"\bGeschäftsführerin\b",
    r"\bLeiter von\b",
    r"\bLeitung\b",
    r"\bVizepräsident\b",
    r"\bChef\b",
    r"\bVorstandsvorsitzender\b",
    r"\bFinanzvorstand\b",
    r"\bFinanzdirektor\b",
    r"\bTechnischer Leiter\b",
    r"\bTechnologievorstand\b",
    r"\bBetriebsleiter\b",
    r"\bTeamleiter\b",
    r"\bGesellschafter\b",
    r"\bFührungskraft\b",
    r"\bLeitender Angestellter\b",
    r"\bGründer\b",
    r"\bGründerin\b",
    r"\bPräsident\b",
    r"\bInhaber\b",
    r"\bInhaberin\b",
    r"\bGruppenleiter\b",
    r"\bVorsitzender\b",
    r"\bMitbegründer\b",
]
LEADERSHIP_PATTERN = re.compile("|".join(LEADERSHIP_KEYWORDS), re.IGNORECASE)
LEADERSHIP_MGMT_LEVELS = {
    "manager",
    "director",
    "vp",
    "executive",
    "partner",
    "manager",
    "vice president",
    "president/vice president",
    "head",
    "owner",
    "founder",
    "c-level",
}


def get_mongo_collections():
    """Connects to MongoDB and returns the required collection objects."""
    log.info("Connecting to MongoDB...")
    try:
        client_url = os.getenv("MONGO_URI")
        db_archive_name = os.getenv("MONGO_DB_ARCHIVE")
        db_latest_name = os.getenv("MONGO_DB")
        col_people_name = os.getenv("PEOPLE_COLLECTION")
        col_companies_name = os.getenv("COMPANY_COLLECTION")
        col_output_name = os.getenv("LEADERSHIP_CHANGE_COLLECTION")

        client = pymongo.MongoClient(client_url)
        client.server_info()

        db_archive = client[db_archive_name]
        db_leadcruit = client[db_latest_name]

        collections = {
            "base_people": db_archive[col_people_name],
            "latest_people": db_leadcruit[col_people_name],
            "companies": db_leadcruit[col_companies_name],
            "output": db_leadcruit[col_output_name],
        }
        log.info("MongoDB connection successful.")
        return collections
    except pymongo.errors.ConnectionFailure as e:
        log.error(f"MongoDB connection failed: {e}")
        return None


def get_analysis_dates(base_people_col):
    """
    Finds the latest source_date in the archive to set the base period
    and calculates the latest period (base + 1 month).
    """
    log.info("Finding latest source_date in leadcruit_archive...")
    try:
        latest_doc = base_people_col.find_one(
            {"source_date": {"$exists": True}},
            sort=[("source_date", pymongo.DESCENDING)],
        )
        if not latest_doc:
            log.error("No documents with 'source_date' found in archive.")
            return None, None

        base_date = latest_doc["source_date"]
        base_date = datetime.datetime(base_date.year, base_date.month, 1)

        latest_date = base_date + relativedelta(months=1)

        log.info(
            f"Base period (max archive date) set to: {base_date.strftime('%Y-%m')}"
        )
        log.info(f"Latest period (target) set to: {latest_date.strftime('%Y-%m')}")

        return base_date, latest_date

    except Exception as e:
        log.error(f"Error finding analysis dates: {e}")
        return None, None


def fetch_people_data(collection, filter_date):
    """
    Fetches people data from a collection for a specific month and year.
    """
    year = filter_date.year
    month = filter_date.month
    log.info(f"Fetching data from {collection.full_name} for {year}-{month:02d}...")

    start_of_month = datetime.datetime(year, month, 1)
    end_of_month = start_of_month + relativedelta(months=1)

    query = {"source_date": {"$gte": start_of_month, "$lt": end_of_month}}

    projection = {
        "member_id": 1,
        "member_full_name": 1,
        "member_websites_linkedin": 1,
        "company_id": 1,
        "member_job_title": 1,
        "management_level": 1,
        "member_experience": 1,
        "_id": 0,
    }

    try:
        count = collection.count_documents(query)
        if count == 0:
            log.warning(
                f"No documents found for {year}-{month:02d} in {collection.full_name}."
            )
            return pd.DataFrame()

        cursor = collection.find(query, projection)
        df = pd.DataFrame(tqdm(cursor, total=count, desc=f"Loading {collection.name}"))
        log.info(f"Loaded {len(df)} records from {collection.name}.")
        return df
    except Exception as e:
        log.error(f"Failed to fetch data from {collection.name}: {e}")
        return pd.DataFrame()


def clean_size_range(value: Optional[str]) -> Optional[str]:
    """Cleans the company size_range string for database storage."""
    if pd.isna(value) or value is None:
        return None
    if value == "Myself Only":
        return "1"
    value = str(value).replace(" employees", "")
    value = str(value).replace(",", "")
    return value


def fetch_company_data(company_col, company_ids):
    """
    Fetches enrichment data for a list of company IDs.
    Returns a DataFrame indexed by company_id, with cleaned company size.
    """
    if not company_ids:
        log.info("No company IDs to enrich, skipping company data fetch.")
        return pd.DataFrame()

    log.info(f"Fetching enrichment data for {len(company_ids)} companies...")
    query = {"company_id": {"$in": list(company_ids)}}
    projection = {
        "_id": 0,
        "company_id": 1,
        "industry": 1,
        "hq_country": 1,
        "size_range": 1,
    }

    try:
        df_companies = pd.DataFrame(list(company_col.find(query, projection)))

        if df_companies.empty:
            log.warning("No matching company details found for enrichment.")
            return pd.DataFrame()

        df_companies = df_companies.set_index("company_id")

        df_companies["company_size"] = df_companies["size_range"].apply(
            clean_size_range
        )

        log.info(f"Found {len(df_companies)} matching company detail records.")
        return df_companies

    except Exception as e:
        log.error(f"Failed to fetch company enrichment data: {e}")
        return pd.DataFrame()


def analyze_changes(df_sep, df_oct, latest_period):
    """
    Runs the core leadership change analysis logic.
    """
    log.info("Starting analysis...")

    def check_is_leader(title, mgmt_level):
        if pd.isna(mgmt_level) and pd.isna(title):
            return False
        if isinstance(mgmt_level, str) and mgmt_level.lower() in LEADERSHIP_MGMT_LEVELS:
            return True
        if pd.isna(title):
            return False
        if isinstance(title, str) and LEADERSHIP_PATTERN.search(title):
            return True
        return False

    def robust_parse_date(date_str):
        if not date_str or pd.isna(date_str):
            return None
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            try:
                return datetime.datetime.strptime(date_str, "%Y-%m")
            except (ValueError, TypeError):
                return None

    def get_all_active_jobs(experience_list):
        active_jobs = []
        if not isinstance(experience_list, (list, np.ndarray)):
            return active_jobs
        for job in experience_list:
            if not job:
                continue
            date_to = job.get("date_to")
            if date_to is None or pd.isna(date_to):
                title = job.get("title")
                mgmt_level = job.get("management_level")
                date_from_str = job.get("date_from")
                parsed_date_from = robust_parse_date(date_from_str)
                active_jobs.append(
                    {
                        "company_id": job.get("company_id"),
                        "company_name": job.get("company_name", "Unknown"),
                        "title": title,
                        "management_level": mgmt_level,
                        "is_leader": check_is_leader(title, mgmt_level),
                        "date_from": parsed_date_from,
                    }
                )
        return active_jobs

    log.info("Merging base and latest people data...")
    df_merged = pd.merge(
        df_sep, df_oct, on="member_id", how="outer", suffixes=("_sep", "_oct")
    )
    log.info(f"Merged DataFrame has {len(df_merged)} total unique members.")
    del df_sep, df_oct

    change_events = []

    ANALYSIS_YEAR = latest_period["year"]
    ANALYSIS_MONTH = latest_period["month"]
    log.info(f"Analyzing changes for {ANALYSIS_YEAR}-{ANALYSIS_MONTH:02d}")

    for row in tqdm(
        df_merged.itertuples(index=False),
        total=len(df_merged),
        desc="Processing members",
    ):
        exp_sep = row.member_experience_sep
        exp_oct = row.member_experience_oct
        is_sep_data_valid_and_not_empty = (
            isinstance(exp_sep, (list, np.ndarray)) and len(exp_sep) > 0
        )
        is_oct_data_invalid_or_empty = (
            not isinstance(exp_oct, (list, np.ndarray)) or len(exp_oct) == 0
        )
        if is_sep_data_valid_and_not_empty and is_oct_data_invalid_or_empty:
            continue

        active_jobs_sep = get_all_active_jobs(exp_sep)
        active_jobs_oct = get_all_active_jobs(exp_oct)

        sep_leader_map = {}
        for job in active_jobs_sep:
            if pd.isna(job["company_id"]):
                continue
            company_id = job["company_id"]
            current_status = sep_leader_map.get(company_id, (False, "Unknown"))[0]
            new_status = current_status or job["is_leader"]
            sep_leader_map[company_id] = (new_status, job["company_name"])

        oct_leader_map = {}
        for job in active_jobs_oct:
            if pd.isna(job["company_id"]):
                continue
            company_id = job["company_id"]
            current_status = oct_leader_map.get(company_id, (False, "Unknown"))[0]
            new_status = current_status or job["is_leader"]
            oct_leader_map[company_id] = (new_status, job["company_name"])

        sep_co_ids = set(sep_leader_map.keys())
        oct_co_ids = set(oct_leader_map.keys())

        new_co_ids = oct_co_ids - sep_co_ids
        left_co_ids = sep_co_ids - oct_co_ids
        common_co_ids = sep_co_ids & oct_co_ids

        member_name = (
            row.member_full_name_oct
            if pd.notna(row.member_full_name_oct)
            else row.member_full_name_sep
        )
        linkedin_url = (
            row.member_websites_linkedin_oct
            if pd.notna(row.member_websites_linkedin_oct)
            else row.member_websites_linkedin_sep
        )
        member_details = {
            "member_id": row.member_id,
            "member_name": member_name if pd.notna(member_name) else "Unknown",
            "linkedin_url": linkedin_url if pd.notna(linkedin_url) else "Unknown",
        }

        for company_id in new_co_ids:
            is_leader, company_name = oct_leader_map[company_id]
            if is_leader:
                is_recent = False
                for job in active_jobs_oct:
                    if job["company_id"] == company_id and job["is_leader"]:
                        if job["date_from"]:
                            if (
                                job["date_from"].year == ANALYSIS_YEAR
                                and job["date_from"].month == ANALYSIS_MONTH
                            ):
                                is_recent = True
                                break
                if is_recent:
                    change_events.append(
                        {
                            "company_id": company_id,
                            "company_name": company_name,
                            "change_type": "leader_joined_from_outside",
                            "member": member_details,
                        }
                    )

        for company_id in left_co_ids:
            is_leader, company_name = sep_leader_map[company_id]
            if is_leader:
                change_events.append(
                    {
                        "company_id": company_id,
                        "company_name": company_name,
                        "change_type": "leader_moved_out",
                        "member": member_details,
                    }
                )

        for company_id in common_co_ids:
            is_leader_sep, name_sep = sep_leader_map[company_id]
            is_leader_oct, name_oct = oct_leader_map[company_id]
            company_name = name_oct if name_oct != "Unknown" else name_sep

            if not is_leader_sep and is_leader_oct:
                change_events.append(
                    {
                        "company_id": company_id,
                        "company_name": company_name,
                        "change_type": "leader_promoted_internally",
                        "member": member_details,
                    }
                )
            elif is_leader_sep and not is_leader_oct:
                change_events.append(
                    {
                        "company_id": company_id,
                        "company_name": company_name,
                        "change_type": "leader_moved_out",
                        "member": member_details,
                    }
                )

    log.info(
        f"Finished processing. Found {len(change_events)} individual leadership change events."
    )

    log.info("Aggregating results by company...")
    company_changes = defaultdict(
        lambda: {
            "leadership_change": True,
            "company_name": "Unknown",
            "leader_moved_out": [],
            "leader_joined_from_outside": [],
            "leader_promoted_internally": [],
        }
    )
    for event in change_events:
        company_id = event["company_id"]
        if not company_id or pd.isna(company_id):
            continue
        company_changes[company_id]["company_name"] = event["company_name"]
        company_changes[company_id][event["change_type"]].append(event["member"])

    log.info(f"Found {len(company_changes)} companies with leadership changes.")
    return company_changes


def save_results_to_mongo(company_changes, df_companies, output_col):
    """
    Enriches the aggregated changes with company data and saves to MongoDB.
    """
    log.info(f"Enriching and saving results to {output_col.full_name}...")

    company_data_map = {}
    if not df_companies.empty:
        company_data_map = df_companies.to_dict("index")

    try:
        output_documents = []
        for company_id, data in company_changes.items():
            enrichment = company_data_map.get(company_id, {})

            doc = {
                "company_id": int(company_id),
                "company_name": data["company_name"],
                "industry": enrichment.get("industry"),
                "hq_country": enrichment.get("hq_country"),
                "company_size": enrichment.get("company_size"),
                "leadership_change": data["leadership_change"],
                "leaders_moved_out": data["leader_moved_out"],
                "leaders_joined_from_outside": data["leader_joined_from_outside"],
                "leaders_promoted_internally": data["leader_promoted_internally"],
                "last_updated": datetime.datetime.now(),
            }
            output_documents.append(doc)

        if output_documents:
            log.info(f"Clearing old data from {output_col.full_name}...")
            output_col.delete_many({})
            log.info(
                f"Inserting {len(output_documents)} new company analysis documents."
            )
            output_col.insert_many(output_documents)
        else:
            log.info("No leadership changes found to save.")

        log.info("Analysis complete and data saved.")

        log.info("\n--- Example Output Document ---")
        example = output_col.find_one()
        if example:
            log.info(json.dumps(example, indent=2, default=str))
        else:
            log.info("No output documents were generated.")

    except Exception as e:
        log.error(f"Error saving data to MongoDB: {e}")


def main():
    """
    Main function to run the complete ETL process.
    """
    log.info("--- Starting Leadership Change ETL Process ---")

    collections = get_mongo_collections()
    if not collections:
        sys.exit(1)

    base_date, latest_date = get_analysis_dates(collections["base_people"])
    if not base_date or not latest_date:
        log.error("Could not determine analysis dates. Exiting.")
        sys.exit(1)

    latest_period = {"year": latest_date.year, "month": latest_date.month}

    df_base_people = fetch_people_data(collections["base_people"], base_date)
    df_latest_people = fetch_people_data(collections["latest_people"], latest_date)

    if df_base_people.empty or df_latest_people.empty:
        log.error("Missing data for one or both periods. Cannot compare. Exiting.")
        sys.exit(1)

    company_changes = analyze_changes(df_base_people, df_latest_people, latest_period)

    if not company_changes:
        log.info("No company changes found. Exiting.")
        sys.exit(0)

    all_company_ids = list(company_changes.keys())
    df_companies = fetch_company_data(collections["companies"], all_company_ids)

    save_results_to_mongo(company_changes, df_companies, collections["output"])

    log.info("--- Leadership Change ETL Process Finished ---")


if __name__ == "__main__":
    main()

# python3 leadership_changes.py
