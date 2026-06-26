#!/usr/bin/env python3
"""
KickCast Pipeline Glue Script
==============================
Connects Component 1 (Live API Fetching) with Component 2 (ML Predictions)
and saves the output to Azure Table Storage.

Steps:
1. Connects to Azure Table Storage (Azurite by default).
2. Fetches latest World Cup matches from football-data.org.
3. Loads the trained ML model and Elo ratings.
4. Generates predictions for upcoming (SCHEDULED/TIMED) matches.
5. Upserts predictions to the 'PredictionsTable'.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import load_settings
from data.football_api import FootballAPIClient
from data.storage import TableStore, BlobStore
from model.predict import load_prediction_context, predict_fixtures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def run_pipeline() -> None:
    settings = load_settings()

    logger.info("Starting KickCast automation pipeline...")

    # Initialize dependencies
    blob_store = BlobStore(settings.azure_storage_connection_string)
    api_client = FootballAPIClient(
        base_url=settings.football_api_base_url,
        api_key=settings.football_api_key,
        competition=settings.football_api_competition,
        blob_store=blob_store,
        cache_container=settings.blob_container_raw,
    )
    table_store = TableStore(settings.azure_storage_connection_string)

    # 1. Fetch upcoming World Cup matches
    logger.info("Fetching WC 2026 matches...")
    try:
        matches_data = api_client.fetch_matches()
        if matches_data is None:
            logger.info("No match data returned from API.")
            return
    except Exception as e:
        logger.error(f"Failed to fetch matches: {e}")
        # If we had caching from blob store implemented in api client,
        # we might fallback here. For now, raise.
        raise

    from data.football_api import extract_matches
    matches = extract_matches(matches_data)
    logger.info(f"Retrieved {len(matches)} matches total.")

    # Filter for valid matches (must have both teams)
    valid_matches = [
        m for m in matches 
        if m.get("home_team") is not None and m.get("home_team") != ""
        and m.get("away_team") is not None and m.get("away_team") != ""
    ]
    logger.info(f"Found {len(valid_matches)} valid matches.")

    if not valid_matches:
        logger.info("No valid matches found. Pipeline complete.")
        return

    # 2. Load Prediction Context
    model_path = str(_project_root / "models" / "kickcast_model.joblib")
    if not Path(model_path).exists():
        logger.error("Model not found. Run train_model.py first.")
        sys.exit(1)

    logger.info("Loading Prediction Context...")
    ctx = load_prediction_context(
        model_path=model_path,
        historical_csv_path=settings.historical_data_path,
    )

    # 2.5 Update Elo with recent tournament results
    finished_matches = [
        m for m in valid_matches 
        if m.get("status") in ["FINISHED", "IN_PLAY"] and m.get("home_score", -1) >= 0
    ]
    if finished_matches:
        logger.info(f"Dynamically updating Elo ratings and history using {len(finished_matches)} recent matches...")
        from model.predict import update_elo_with_recent_results, update_match_history_with_recent_results
        update_elo_with_recent_results(ctx, finished_matches)
        update_match_history_with_recent_results(ctx, finished_matches)

    # 3. Predict Outcomes
    logger.info("Generating predictions...")
    fixtures_for_prediction = []
    for m in valid_matches:
        fixtures_for_prediction.append({
            "id": m.get("match_id"),
            "date": m.get("utc_date"),
            "home_team": m.get("home_team"),
            "away_team": m.get("away_team"),
            "is_neutral": True,  # WC matches are neutral
            "stage": m.get("stage", "Group Stage")
        })

    predictions = predict_fixtures(fixtures_for_prediction, ctx)
    
    # 4. Save to Azure Table
    table_name = "predictions"
    logger.info(f"Upserting {len(predictions)} predictions to Table: {table_name}")
    
    for i, pred in enumerate(predictions):
        match_id = str(fixtures_for_prediction[i]["id"])
        fixture = valid_matches[i]
        
        entity = {
            "PartitionKey": "wc2026",
            "RowKey": match_id,
            "Date": fixtures_for_prediction[i]["date"],
            "Stage": fixture.get("stage", "World Cup"),
            "Status": fixture.get("status", ""),
            "ActualHomeScore": fixture.get("home_score", -1),
            "ActualAwayScore": fixture.get("away_score", -1),
            "HomePenScore": fixture.get("home_pen_score"),
            "AwayPenScore": fixture.get("away_pen_score"),
            "IsAet": fixture.get("is_aet", False),
            "HomeScorers": fixture.get("home_scorers", ""),
            "AwayScorers": fixture.get("away_scorers", ""),
            "HomeTeam": pred["home_team"],
            "AwayTeam": pred["away_team"],
            "HomeElo": float(pred["home_elo"]),
            "AwayElo": float(pred["away_elo"]),
            "EloDiff": float(pred["elo_diff"]),
            "HomeWinProb": float(pred["home_win"]),
            "DrawProb": float(pred["draw"]),
            "AwayWinProb": float(pred["away_win"]),
            "PredictedOutcome": pred["predicted_outcome"],
            "Confidence": float(pred["confidence"]),
            "PredictedHomeScore": float(pred.get("predicted_home_score", 0)),
            "PredictedAwayScore": float(pred.get("predicted_away_score", 0)),
            "AdvanceMethod": pred.get("advance_method", ""),
            "AetHomeProb": float(pred.get("aet_home_prob", 0.0)),
            "AetAwayProb": float(pred.get("aet_away_prob", 0.0)),
            "PenHomeProb": float(pred.get("pen_home_prob", 0.0)),
            "PenAwayProb": float(pred.get("pen_away_prob", 0.0)),
            "UpdatedAt": datetime.now(timezone.utc).isoformat()
        }
        table_store.upsert_entity(table_name, entity)

    logger.info("Pipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
