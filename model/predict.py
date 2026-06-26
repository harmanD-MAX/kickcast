"""
Prediction Interface
====================
Provides the function that the PowerShell pipeline calls to get
predictions for upcoming World Cup fixtures.

This is the public API of the model package. It takes a fixture
(home team, away team, neutral venue flag) and returns calibrated
win/draw/loss probabilities.

Usage:
    from model.predict import predict_match, load_prediction_context

    ctx = load_prediction_context(model_path, historical_csv_path)
    prediction = predict_match("Brazil", "Germany", is_neutral=True, ctx=ctx)
    print(prediction)
    # {'home_win': 0.45, 'draw': 0.28, 'away_win': 0.27, 'predicted_outcome': 'H'}
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from model.elo import EloRatingSystem
from model.features import (
    compute_features_for_fixture,
    RESULT_LABELS,
)
from model.train import load_model
from data.historical import load_historical_data

logger = logging.getLogger(__name__)


@dataclass
class PredictionContext:
    """
    Holds everything needed to make predictions:
    - The trained model
    - The score models
    - Current Elo ratings
    - Recent match history (for form/goals features)
    """
    model: CalibratedClassifierCV
    score_models: dict
    elo_ratings: dict[str, float]
    match_history: pd.DataFrame


def load_prediction_context(
    model_path: str,
    historical_csv_path: str,
    score_model_path: str = None,
) -> PredictionContext:
    """
    Load the model and compute the Elo ratings and match history
    needed for predictions.

    This is called once at startup, not per-prediction.

    Args:
        model_path: Path to the saved model (joblib file)
        historical_csv_path: Path to the historical CSV
        score_model_path: Path to the score model

    Returns:
        A PredictionContext with model, Elo ratings, and history
    """
    import joblib
    from pathlib import Path
    
    # Load trained model
    model = load_model(model_path)
    
    if score_model_path is None:
        score_model_path = str(Path(model_path).parent / "score_model.joblib")
    
    try:
        score_models = joblib.load(score_model_path)
    except FileNotFoundError:
        logger.warning(f"Score model not found at {score_model_path}")
        score_models = None

    # Load and process historical data to get Elo ratings
    df = load_historical_data(historical_csv_path)

    # Compute Elo ratings from full history
    elo_system = EloRatingSystem()
    df = elo_system.compute_all_ratings(df)

    # Keep recent matches for form/goals computation
    # (Last 100 matches per team should be plenty)
    match_history = df.tail(5000)  # Recent matches for all teams

    return PredictionContext(
        model=model,
        score_models=score_models,
        elo_ratings=elo_system.get_ratings(),
        match_history=match_history,
    )


def predict_match(
    home_team: str,
    away_team: str,
    is_neutral: bool,
    ctx: PredictionContext,
    stage: str = "Group Stage",
) -> dict:
    """
    Predict the outcome probabilities for a single fixture.

    Args:
        home_team: Name of the home team (must match names in dataset)
        away_team: Name of the away team
        is_neutral: Whether the venue is neutral
        ctx: PredictionContext with model, Elo ratings, and history

    Returns:
        dict with keys:
          - home_win: float (probability)
          - draw: float (probability)
          - away_win: float (probability)
          - predicted_outcome: str ('H', 'D', or 'A')
          - home_team: str
          - away_team: str
          - home_elo: float
          - away_elo: float
          - elo_diff: float
          - confidence: float (probability of predicted outcome)
    """
    # Compute features for this fixture
    X = compute_features_for_fixture(
        home_team=home_team,
        away_team=away_team,
        is_neutral=is_neutral,
        elo_ratings=ctx.elo_ratings,
        match_history=ctx.match_history,
        stage=stage,
    )

    # Get calibrated probabilities
    proba = ctx.model.predict_proba(X)[0]

    # Map to class labels
    # proba order matches model.classes_ — typically [0, 1, 2] = [H, D, A]
    home_win_prob = float(proba[0])
    draw_prob = float(proba[1])
    away_win_prob = float(proba[2])

    # Predicted outcome is the class with highest probability
    predicted_class = int(np.argmax(proba))
    predicted_outcome = RESULT_LABELS[predicted_class]
    confidence = float(proba[predicted_class])

    # Removed Draw Heuristic as requested by user to allow decisive predictions

    home_elo = ctx.elo_ratings.get(home_team, 1500.0)
    away_elo = ctx.elo_ratings.get(away_team, 1500.0)

    advance_method = "Regulation"
    pred_aet_home_score = None
    pred_aet_away_score = None
    pred_pen_home_score = None
    pred_pen_away_score = None

    # Predict exact score
    pred_home_score = 0
    pred_away_score = 0
    if ctx.score_models is not None:
        from model.score_model import predict_exact_score
        score_preds = predict_exact_score(
            ctx.score_models['home_model'], 
            ctx.score_models['away_model'], 
            X
        )
        if score_preds:
            pred_home_score = score_preds[0]['home_score']
            pred_away_score = score_preds[0]['away_score']
            
            # Align predicted outcome with exact score prediction
            if pred_home_score > pred_away_score:
                predicted_outcome = 'H'
                confidence = max(confidence, home_win_prob)
            elif pred_home_score < pred_away_score:
                predicted_outcome = 'A'
                confidence = max(confidence, away_win_prob)
            else:
                if "Group" not in stage and stage != "Group Stage":
                    total_win = home_win_prob + away_win_prob
                    if total_win == 0: total_win = 1
                    
                    home_ratio = home_win_prob / total_win
                    away_ratio = away_win_prob / total_win
                    
                    if home_ratio > 0.52:
                        pred_aet_home_score = 1
                        pred_aet_away_score = 0
                        predicted_outcome = 'H'
                        advance_method = "Extra Time"
                        confidence = max(confidence, home_win_prob)
                    elif away_ratio > 0.52:
                        pred_aet_home_score = 0
                        pred_aet_away_score = 1
                        predicted_outcome = 'A'
                        advance_method = "Extra Time"
                        confidence = max(confidence, away_win_prob)
                    else:
                        pred_aet_home_score = 0
                        pred_aet_away_score = 0
                        advance_method = "Penalties"
                        
                        if home_elo > away_elo:
                            pred_pen_home_score = 5
                            pred_pen_away_score = 4
                            predicted_outcome = 'H'
                        else:
                            pred_pen_home_score = 4
                            pred_pen_away_score = 5
                            predicted_outcome = 'A'
                else:
                    predicted_outcome = 'D'
                    confidence = max(confidence, draw_prob)
                    advance_method = ""

    # Compute AET/Pen probabilities for knockout stages
    aet_home_prob = 0.0
    aet_away_prob = 0.0
    pen_home_prob = 0.0
    pen_away_prob = 0.0
    if stage != "Group Stage":
        # Distribute the draw probability into AET and Penalties based on Elo
        total_elo = home_elo + away_elo
        if total_elo == 0: total_elo = 1
        home_edge = home_elo / total_elo
        away_edge = away_elo / total_elo
        
        # Assume 40% of draws end in AET, 60% go to penalties
        aet_prob = draw_prob * 0.4
        pen_prob = draw_prob * 0.6
        
        aet_home_prob = aet_prob * home_edge
        aet_away_prob = aet_prob * away_edge
        pen_home_prob = pen_prob * home_edge
        pen_away_prob = pen_prob * away_edge

    return {
        "home_win": round(home_win_prob, 4),
        "draw": round(draw_prob, 4),
        "away_win": round(away_win_prob, 4),
        "predicted_outcome": predicted_outcome,
        "home_team": home_team,
        "away_team": away_team,
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "elo_diff": round(home_elo - away_elo, 1),
        "confidence": round(confidence, 4),
        "predicted_home_score": pred_home_score,
        "predicted_away_score": pred_away_score,
        "advance_method": advance_method,
        "pred_aet_home_score": pred_aet_home_score,
        "pred_aet_away_score": pred_aet_away_score,
        "pred_pen_home_score": pred_pen_home_score,
        "pred_pen_away_score": pred_pen_away_score,
        "aet_home_prob": round(aet_home_prob, 4),
        "aet_away_prob": round(aet_away_prob, 4),
        "pen_home_prob": round(pen_home_prob, 4),
        "pen_away_prob": round(pen_away_prob, 4),
    }


def predict_fixtures(
    fixtures: list[dict],
    ctx: PredictionContext,
) -> list[dict]:
    """
    Predict outcomes for a batch of fixtures.

    Args:
        fixtures: List of dicts with keys: home_team, away_team, is_neutral
        ctx: PredictionContext

    Returns:
        List of prediction dicts (same as predict_match output)
    """
    results = []
    for fixture in fixtures:
        try:
            pred = predict_match(
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                is_neutral=fixture.get("is_neutral", True),  # WC = neutral
                ctx=ctx,
                stage=fixture.get("stage", "Group Stage"),
            )
            results.append(pred)
        except Exception as e:
            logger.error(
                "Failed to predict %s vs %s: %s",
                fixture.get("home_team"),
                fixture.get("away_team"),
                e,
            )
    return results


def update_elo_with_recent_results(ctx: PredictionContext, finished_matches: list[dict]) -> None:
    """
    Updates the Elo ratings in the PredictionContext using the actual results
    of recently finished tournament matches.
    """
    elo_system = EloRatingSystem()
    # Seed with current ratings
    elo_system._ratings = ctx.elo_ratings.copy()
    
    for match in finished_matches:
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        home_score = match.get("home_score", -1)
        away_score = match.get("away_score", -1)
        
        if home_score >= 0 and away_score >= 0 and home_team and away_team:
            elo_system.update(
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                tournament="FIFA World Cup"
            )
            
    # Save back
    ctx.elo_ratings = elo_system.get_ratings()

def update_match_history_with_recent_results(ctx: PredictionContext, finished_matches: list[dict]) -> None:
    """
    Appends recently finished tournament matches to the match history dataframe
    so feature extraction logic uses the latest tournament form, H2H, and goals data.
    """
    new_rows = []
    for match in finished_matches:
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        home_score = match.get("home_score", -1)
        away_score = match.get("away_score", -1)
        date = match.get("date", "")
        stage = match.get("stage", "Group Stage")
        
        if home_score >= 0 and away_score >= 0 and home_team and away_team:
            # Derive result from scores
            if home_score > away_score:
                result = "H"
            elif home_score < away_score:
                result = "A"
            else:
                result = "D"

            new_rows.append({
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "FIFA World Cup",
                "neutral": True,
                "result": result,
                "stage": stage,
            })
            
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        ctx.match_history = pd.concat([ctx.match_history, new_df], ignore_index=True)
        logger.info("Appended %d recent WC match results to match history for form/H2H computation.", len(new_rows))
