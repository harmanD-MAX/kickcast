"""
Historical Data Loader
======================
Loads and preprocesses the Kaggle international football results dataset
(martj42/international-football-results-from-1872-to-2017).

The dataset CSV has columns:
  date, home_team, away_team, home_score, away_score,
  tournament, city, country, neutral

This module:
  1. Reads the CSV from the path configured in Settings.
  2. Validates the expected schema — fails loudly if columns are missing.
  3. Parses dates, drops rows with missing scores.
  4. Derives a `result` column from the scores:
       H = home win, A = away win, D = draw.
  5. Optionally uploads the raw CSV to Azure Blob Storage for archival.
  6. Returns a clean pandas DataFrame ready for feature engineering.

All team names, scores, and dates come from the CSV — nothing is
hardcoded or invented in this module.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from data.storage import BlobStore

logger = logging.getLogger(__name__)

# The exact columns we expect in the Kaggle CSV.
EXPECTED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}


def load_historical_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the historical results CSV.

    Args:
        csv_path: Path to the results.csv file (from Settings).

    Returns:
        A pandas DataFrame with columns:
          date (datetime), home_team, away_team, home_score (int),
          away_score (int), tournament, city, country, neutral (bool),
          result (str: H/A/D), goal_diff (int: home_score - away_score)

    Raises:
        FileNotFoundError: If the CSV doesn't exist.
        ValueError: If expected columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Historical data CSV not found at: {path}\n"
            f"Download it from: https://www.kaggle.com/datasets/"
            f"martj42/international-football-results-from-1872-to-2017\n"
            f"Place it at the path configured in HISTORICAL_DATA_PATH."
        )

    logger.info("Loading historical data from: %s", path)
    df = pd.read_csv(path)

    # --- Schema validation ---
    actual_columns = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_columns
    if missing:
        raise ValueError(
            f"Historical CSV is missing expected columns: {missing}\n"
            f"Found columns: {sorted(actual_columns)}\n"
            f"Expected: {sorted(EXPECTED_COLUMNS)}"
        )

    original_count = len(df)

    # --- Parse dates ---
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # --- Drop rows with missing scores (matches not yet played, or data gaps) ---
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    dropped = original_count - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with missing scores", dropped)

    # --- Parse neutral venue flag ---
    # The CSV uses TRUE/FALSE strings or boolean values
    df["neutral"] = df["neutral"].astype(str).str.upper().map(
        {"TRUE": True, "FALSE": False}
    ).fillna(False)

    # --- Derive result column from actual scores ---
    # H = home win, A = away win, D = draw
    # This is computed, never hardcoded.
    df["result"] = df.apply(_derive_result, axis=1)

    # --- Derive goal difference (useful feature for ML) ---
    df["goal_diff"] = df["home_score"] - df["away_score"]

    # --- Sort by date ---
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Loaded %d historical matches (date range: %s to %s)",
        len(df),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    # Log result distribution
    result_counts = df["result"].value_counts()
    logger.info(
        "Result distribution: H=%d (%.1f%%), D=%d (%.1f%%), A=%d (%.1f%%)",
        result_counts.get("H", 0),
        100 * result_counts.get("H", 0) / len(df),
        result_counts.get("D", 0),
        100 * result_counts.get("D", 0) / len(df),
        result_counts.get("A", 0),
        100 * result_counts.get("A", 0) / len(df),
    )

    return df


def archive_to_blob(
    csv_path: str,
    blob_store: BlobStore,
    container_name: str,
    blob_name: str = "historical/results.csv",
) -> None:
    """
    Upload the raw historical CSV to Azure Blob Storage for archival.

    This preserves the original data alongside any processed versions,
    supporting reproducibility.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Cannot archive — CSV not found at %s", path)
        return

    with open(path, "rb") as f:
        raw_bytes = f.read()

    blob_store.upload_blob(container_name, blob_name, raw_bytes)
    logger.info(
        "Archived historical CSV to blob: %s/%s (%d bytes)",
        container_name,
        blob_name,
        len(raw_bytes),
    )


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for logging / display.
    All numbers are computed from the data, never invented.
    """
    result_counts = df["result"].value_counts()
    tournaments = df["tournament"].nunique()
    teams = pd.concat([df["home_team"], df["away_team"]]).nunique()

    return {
        "total_matches": len(df),
        "date_range_start": str(df["date"].min().date()),
        "date_range_end": str(df["date"].max().date()),
        "unique_teams": teams,
        "unique_tournaments": tournaments,
        "home_wins": int(result_counts.get("H", 0)),
        "draws": int(result_counts.get("D", 0)),
        "away_wins": int(result_counts.get("A", 0)),
        "home_win_pct": round(100 * result_counts.get("H", 0) / len(df), 1),
        "draw_pct": round(100 * result_counts.get("D", 0) / len(df), 1),
        "away_win_pct": round(100 * result_counts.get("A", 0) / len(df), 1),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_result(row: pd.Series) -> str:
    """Derive match result from scores. Never hardcoded."""
    if row["home_score"] > row["away_score"]:
        return "H"
    elif row["home_score"] < row["away_score"]:
        return "A"
    else:
        return "D"
