#!/usr/bin/env python3
"""
KickCast Model Training & Backtest
===================================
End-to-end script that:
  1. Loads historical data
  2. Computes Elo ratings for all teams
  3. Engineers features
  4. Splits chronologically (train < split_date, test >= split_date)
  5. Trains the model
  6. Runs backtest + baseline evaluation
  7. Prints measured results (never invented)
  8. Saves the trained model

Usage:
  cd kickcast
  source .venv/bin/activate
  python scripts/train_model.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import load_settings
from data.historical import load_historical_data, get_dataset_summary
from model.elo import EloRatingSystem
from model.features import build_features, FEATURE_COLUMNS, RESULT_LABELS
from model.train import train_model, save_model
from model.backtest import evaluate_model, evaluate_baseline, print_comparison
from model.predict import predict_match, PredictionContext

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Chronological split date
# ---------------------------------------------------------------------------
# Train on everything before this date, test on everything after.
# June 2024 gives ~2 years of test data including recent WC qualifiers
# and the current World Cup 2026 group stage.
SPLIT_DATE = "2024-06-01"


def main():
    print("=" * 60)
    print("  KickCast — Model Training & Backtest")
    print("=" * 60)
    print()

    settings = load_settings()

    # ------------------------------------------------------------------
    # 1. Load historical data
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Step 1: Loading historical data")
    print("-" * 60)

    df = load_historical_data(settings.historical_data_path)
    summary = get_dataset_summary(df)
    print(f"  Loaded {summary['total_matches']:,} matches "
          f"({summary['date_range_start']} → {summary['date_range_end']})")
    print()

    # ------------------------------------------------------------------
    # 2. Compute Elo ratings
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Step 2: Computing Elo ratings")
    print("-" * 60)

    elo_system = EloRatingSystem()
    df = elo_system.compute_all_ratings(df)

    # Show top and bottom Elo teams
    ratings = elo_system.get_ratings()
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    print("  Top 15 teams by Elo:")
    for team, elo in sorted_ratings[:15]:
        print(f"    {int(elo):>5d}  {team}")
    print()
    print("  Bottom 5 teams by Elo:")
    for team, elo in sorted_ratings[-5:]:
        print(f"    {int(elo):>5d}  {team}")
    print()

    # ------------------------------------------------------------------
    # 3. Engineer features
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Step 3: Engineering features")
    print("-" * 60)

    df, X, y = build_features(df)
    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Features: {', '.join(FEATURE_COLUMNS)}")
    print()

    # ------------------------------------------------------------------
    # 4. Chronological split
    # ------------------------------------------------------------------
    # y is already returned from build_features (H/D/A targets)
    
    # Create target arrays for Poisson regression
    y_home = df["home_score"].values
    y_away = df["away_score"].values

    # 4. Chronological Train/Test split
    # For international football, chronologically splitting prevents future leakage.
    split_date = pd.to_datetime("2024-06-01")
    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    X_train = X[train_mask]
    y_train = y[train_mask]
    y_home_train = y_home[train_mask]
    y_away_train = y_away[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]

    print("-" * 60)
    print("  Step 4: Chronological split at 2024-06-01")
    print("-" * 60)
    print(f"  Training set: {len(X_train):,} matches (before 2024-06-01)")
    print(f"  Test set:     {len(X_test):,} matches (from 2024-06-01 onward)")

    # 5. Train Model
    print("-" * 60)
    print("  Step 5: Training models")
    print("-" * 60)

    # Compute time-decay sample weights (recent matches count more)
    from model.train import compute_time_decay_weights
    train_dates = df[train_mask]["date"].values
    sample_weights = compute_time_decay_weights(train_dates, half_life_years=5.0)
    print(f"  Time-decay weights computed (half-life: 5 years)")
    
    # Train Gradient Boosting Win/Draw/Loss model with tuning + weights
    model = train_model(X_train, y_train, sample_weights=sample_weights, tune_hyperparams=True)
    print("  Gradient Boosting model trained successfully.")
    
    # Train Poisson Score models
    score_model_path = str(_project_root / "models" / "score_model.joblib")
    from model.score_model import train_score_models
    train_score_models(X_train, y_home_train, y_away_train, score_model_path)
    print("  Poisson Score models trained successfully.")

    # Test set result distribution
    for label_idx, label_name in RESULT_LABELS.items():
        count = np.sum(y_test == label_idx)
        pct = 100 * count / len(y_test) if len(y_test) > 0 else 0
        print(f"    {label_name}: {count} ({pct:.1f}%)")
    print()

    if len(X_test) == 0:
        print("ERROR: No test samples. Adjust SPLIT_DATE.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Backtest
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Step 6: Backtesting")
    print("-" * 60)

    # Model evaluation
    model_results = evaluate_model(model, X_test, y_test, name="KickCast Model")

    # Baseline evaluation
    baseline_results = evaluate_baseline(X_test, y_test)

    # Print comparison
    print_comparison(model_results, baseline_results)

    # Also print raw results for recording
    print(model_results)
    print(baseline_results)

    # ------------------------------------------------------------------
    # 7. Sample predictions on upcoming WC fixtures
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Sample Predictions (using current Elo ratings)")
    print("-" * 60)

    # Get the most recent match history for form computation
    ctx = PredictionContext(
        model=model,
        score_models=None,
        elo_ratings=elo_system.get_ratings(),
        match_history=df.tail(5000),
    )

    # Predict some sample matchups from the dataset's most recent teams
    # (these are real teams from the actual WC 2026 — not hardcoded,
    #  we pull the most recently active teams from the data)
    recent_matches = df[df["tournament"].str.contains("FIFA World Cup", na=False)].tail(10)

    if len(recent_matches) > 0:
        # Get unique teams from recent WC matches
        recent_teams = list(pd.concat([
            recent_matches["home_team"],
            recent_matches["away_team"]
        ]).unique())

        # Make a few predictions between recent WC teams
        print()
        shown = 0
        for i in range(min(len(recent_teams) - 1, 5)):
            for j in range(i + 1, min(len(recent_teams), i + 2)):
                team_a = recent_teams[i]
                team_b = recent_teams[j]
                pred = predict_match(team_a, team_b, is_neutral=True, ctx=ctx)
                print(
                    f"  {pred['home_team']:>25s} vs {pred['away_team']:<25s}"
                )
                print(
                    f"    Elo: {pred['home_elo']:.0f} vs {pred['away_elo']:.0f} "
                    f"(diff: {pred['elo_diff']:+.0f})"
                )
                print(
                    f"    Probabilities:  H={pred['home_win']:.1%}  "
                    f"D={pred['draw']:.1%}  A={pred['away_win']:.1%}"
                )
                print(
                    f"    Prediction: {pred['predicted_outcome']} "
                    f"(confidence: {pred['confidence']:.1%})"
                )
                print()
                shown += 1
                if shown >= 5:
                    break
            if shown >= 5:
                break

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Step 7: Saving model")
    print("-" * 60)

    model_path = str(_project_root / "models" / "kickcast_model.joblib")
    save_model(model, model_path)
    print(f"  Model saved to: {model_path}")

    # Also save Elo ratings
    import json
    elo_path = _project_root / "models" / "elo_ratings.json"
    elo_path.parent.mkdir(parents=True, exist_ok=True)
    with open(elo_path, "w") as f:
        json.dump(
            {team: round(elo, 1) for team, elo in sorted_ratings},
            f, indent=2
        )
    print(f"  Elo ratings saved to: {elo_path}")
    print()

    print("=" * 60)
    print("  Training & backtest complete.")
    print("  All metrics above are measured from real data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
