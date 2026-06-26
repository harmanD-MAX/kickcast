#!/usr/bin/env python3
"""
KickCast Bootstrap Script
=========================
One-time setup that:
  1. Creates Azure Storage containers and tables (in Azurite locally).
  2. Loads the historical CSV into blob storage.
  3. Fetches current World Cup fixtures from the live API.
  4. Writes each fixture to the fixtures table.
  5. Prints a summary — all numbers come from real data.

Prerequisites:
  - Azurite running: npx azurite --location ./azurite_data
  - .env file with FOOTBALL_API_KEY set
  - Historical CSV downloaded to HISTORICAL_DATA_PATH

Usage:
  python scripts/bootstrap_data.py
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path so we can import our modules
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import load_settings
from data.storage import BlobStore, TableStore
from data.football_api import FootballAPIClient, extract_matches, match_to_table_entity
from data.historical import load_historical_data, archive_to_blob, get_dataset_summary

# ---------------------------------------------------------------------------
# Logging — verbose for bootstrap so the user can see what's happening
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bootstrap")


def main():
    print("=" * 60)
    print("  KickCast — Data Layer Bootstrap")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    logger.info("Loading configuration...")
    settings = load_settings()
    logger.info("  API base URL: %s", settings.football_api_base_url)
    logger.info("  Competition: %s", settings.football_api_competition)
    logger.info("  Poll interval: %d seconds", settings.football_api_poll_interval_seconds)
    logger.info("  Historical data: %s", settings.historical_data_path)
    logger.info(
        "  Storage: %s",
        "Azurite (local)" if "127.0.0.1" in settings.azure_storage_connection_string
        else "Azure (remote)",
    )
    print()

    # ------------------------------------------------------------------
    # 2. Initialize storage clients
    # ------------------------------------------------------------------
    logger.info("Initializing storage clients...")
    blob_store = BlobStore(settings.azure_storage_connection_string)
    table_store = TableStore(settings.azure_storage_connection_string)
    print()

    # ------------------------------------------------------------------
    # 3. Load historical data
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Historical Data")
    print("-" * 60)

    historical_ok = False
    try:
        df = load_historical_data(settings.historical_data_path)
        summary = get_dataset_summary(df)

        print(f"  Total matches:     {summary['total_matches']:,}")
        print(f"  Date range:        {summary['date_range_start']} → {summary['date_range_end']}")
        print(f"  Unique teams:      {summary['unique_teams']:,}")
        print(f"  Unique tournaments: {summary['unique_tournaments']:,}")
        print(f"  Home wins:         {summary['home_wins']:,} ({summary['home_win_pct']}%)")
        print(f"  Draws:             {summary['draws']:,} ({summary['draw_pct']}%)")
        print(f"  Away wins:         {summary['away_wins']:,} ({summary['away_win_pct']}%)")
        print()

        # Archive the raw CSV to blob storage
        logger.info("Archiving historical CSV to blob storage...")
        archive_to_blob(
            settings.historical_data_path,
            blob_store,
            settings.blob_container_raw,
        )

        # Also store the summary as JSON
        blob_store.upload_json(
            settings.blob_container_raw,
            "historical/summary.json",
            summary,
        )

        historical_ok = True

    except FileNotFoundError as e:
        logger.warning("Historical data not available: %s", e)
        print("  ⚠ Historical CSV not found. Download it from:")
        print("    https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017")
        print(f"    Place it at: {settings.historical_data_path}")
        print()

    except Exception as e:
        logger.error("Error loading historical data: %s", e)
        print(f"  ✗ Error: {e}")
        print()

    # ------------------------------------------------------------------
    # 4. Fetch live World Cup fixtures
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Live World Cup 2026 Fixtures")
    print("-" * 60)

    api_ok = False
    api_client = FootballAPIClient(
        base_url=settings.football_api_base_url,
        api_key=settings.football_api_key,
        competition=settings.football_api_competition,
        blob_store=blob_store,
        cache_container=settings.blob_container_raw,
    )

    try:
        # Fetch matches
        raw_matches = api_client.fetch_matches()
        if raw_matches is None:
            print("  ✗ Could not fetch matches from API (and no cache available)")
        else:
            matches = extract_matches(raw_matches)
            print(f"  Total fixtures found: {len(matches)}")

            # Count by status
            status_counts: dict[str, int] = {}
            for m in matches:
                s = m["status"]
                status_counts[s] = status_counts.get(s, 0) + 1
            for status, count in sorted(status_counts.items()):
                print(f"    {status}: {count}")
            print()

            # Show a few sample matches (real data, not hardcoded)
            print("  Sample fixtures (first 5):")
            for m in matches[:5]:
                score_str = ""
                if m["home_score"] is not None and m["away_score"] is not None:
                    score_str = f" [{m['home_score']}-{m['away_score']}]"
                date_str = m["utc_date"][:10] if m["utc_date"] else "TBD"
                print(
                    f"    {date_str}  {m['home_team']:>25s} vs {m['away_team']:<25s}"
                    f"  ({m['status']}){score_str}"
                )
            print()

            # Write each match to the fixtures table
            logger.info("Writing %d fixtures to table storage...", len(matches))
            for m in matches:
                entity = match_to_table_entity(m)
                table_store.upsert_entity(settings.table_fixtures, entity)

            stored_count = table_store.count_entities(settings.table_fixtures)
            print(f"  Fixtures stored in table: {stored_count}")
            api_ok = True

        # Also fetch standings
        raw_standings = api_client.fetch_standings()
        if raw_standings:
            standings_groups = raw_standings.get("standings", [])
            print(f"  Group standings fetched: {len(standings_groups)} groups")

        # Also fetch teams
        raw_teams = api_client.fetch_teams()
        if raw_teams:
            teams = raw_teams.get("teams", [])
            print(f"  Teams fetched: {len(teams)}")
            if teams:
                # Show team names (real, from API)
                team_names = [t.get("name", "?") for t in teams]
                print(f"  Participating teams: {', '.join(sorted(team_names))}")

    except Exception as e:
        logger.error("Error fetching live data: %s", e)
        print(f"  ✗ Error: {e}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Bootstrap Summary")
    print("=" * 60)
    print(f"  Historical data:   {'✓ loaded' if historical_ok else '✗ not loaded'}")
    print(f"  Live API data:     {'✓ fetched' if api_ok else '✗ not fetched'}")
    print(f"  Storage backend:   {'Azurite (local)' if '127.0.0.1' in settings.azure_storage_connection_string else 'Azure (remote)'}")
    print()

    if historical_ok and api_ok:
        print("  ✓ Data layer is ready. Proceed to Component 2 (ML model).")
    elif api_ok:
        print("  ⚠ Live data is ready, but historical data is missing.")
        print("    Download the CSV before training the model.")
    elif historical_ok:
        print("  ⚠ Historical data is ready, but live API fetch failed.")
        print("    Check your API key in .env.")
    else:
        print("  ✗ Neither data source is available. Check configuration.")

    print()


if __name__ == "__main__":
    main()
