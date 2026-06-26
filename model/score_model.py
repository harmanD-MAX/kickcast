"""
Score Prediction Model — Poisson Regression
===========================================
Trains a Poisson regression model to predict the expected goals (lambda)
for the home and away teams. We use the same historical features.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

def train_score_models(X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series, output_path: str):
    """
    Train two Poisson Regressors (one for home goals, one for away goals).
    """
    logger.info("Training Poisson Regressors for exact score prediction...")
    
    home_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poisson', PoissonRegressor(alpha=1e-3, max_iter=2000))
    ])
    
    away_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poisson', PoissonRegressor(alpha=1e-3, max_iter=2000))
    ])
    
    logger.info("Fitting home goals model...")
    home_model.fit(X, y_home)
    
    logger.info("Fitting away goals model...")
    away_model.fit(X, y_away)
    
    models = {
        'home_model': home_model,
        'away_model': away_model
    }
    
    joblib.dump(models, output_path)
    logger.info(f"Score models saved to {output_path}")

def predict_exact_score(home_model, away_model, features: pd.DataFrame) -> list[dict]:
    """
    Predicts the expected goals (lambda) and calculates the most probable exact score.
    Returns a list of dicts with 'home_score' and 'away_score'.
    """
    home_lambdas = home_model.predict(features)
    away_lambdas = away_model.predict(features)
    
    predictions = []
    
    for h_lam, a_lam in zip(home_lambdas, away_lambdas):
        # Calculate Poisson probabilities for 0 to 6 goals
        from scipy.stats import poisson
        
        max_prob = -1.0
        best_score = (0, 0)
        
        # Grid search over possible scorelines (0-0 to 6-6)
        for h in range(7):
            for a in range(7):
                prob = poisson.pmf(h, h_lam) * poisson.pmf(a, a_lam)
                if prob > max_prob:
                    max_prob = prob
                    best_score = (h, a)
                    
        predictions.append({
            "home_score": best_score[0],
            "away_score": best_score[1],
            "home_lambda": h_lam,
            "away_lambda": a_lam,
            "probability": max_prob
        })
        
    return predictions
