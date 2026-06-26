"""
Model Training (v3 - Stacking Ensemble)
========================================
Trains a powerful stacking ensemble for multi-class football match prediction.

v3 improvements:
  - StackingClassifier combining XGBoost, LightGBM, and Gradient Boosting
  - Meta-learner: LogisticRegression to optimally weight the base models
  - Isotonic calibration on the stacked ensemble output
  - 22 advanced features including win streaks and goal variance

Why an Ensemble?
  - XGBoost and LightGBM handle non-linear interactions extremely well.
  - GBC adds diversity.
  - The meta-learner smooths out individual model biases.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Import new libraries for v3
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def compute_time_decay_weights(
    dates: np.ndarray,
    half_life_years: float = 5.0,
) -> np.ndarray:
    """
    Compute exponential decay sample weights based on match dates.
    """
    dates = np.array(dates, dtype="datetime64[D]")
    most_recent = dates.max()
    days_ago = (most_recent - dates).astype(float)

    # Exponential decay: w = 2^(-days_ago / half_life_days)
    half_life_days = half_life_years * 365.25
    weights = np.power(2.0, -days_ago / half_life_days)

    # Normalize so the average weight is ~1.0
    weights = weights / weights.mean()

    return weights


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    tune_hyperparams: bool = False,  # Disabled by default for faster stacking
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """
    Train a calibrated Stacking Ensemble.
    """
    logger.info("Training Stacking Ensemble classifier (v3)...")
    logger.info("  Training set: %d samples, %d features", *X_train.shape)

    # --- Base Models ---
    
    # 1. XGBoost (tuned conservatively)
    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=random_state,
        n_jobs=-1,
        # XGBoost handles sample_weight via fit params, but we pass it globally in the stack
    )
    
    # 2. LightGBM (fast and powerful)
    lgb_clf = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    
    # 3. Scikit-Learn Gradient Boosting
    gbc_clf = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        min_samples_leaf=40,
        subsample=0.8,
        random_state=random_state,
    )

    # --- Meta-Learner ---
    # We use logistic regression to optimally weight the predictions of the base models
    meta_clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    # --- Stacking Classifier ---
    stack_clf = StackingClassifier(
        estimators=[
            ("xgb", xgb_clf),
            ("lgb", lgb_clf),
            ("gbc", gbc_clf),
        ],
        final_estimator=meta_clf,
        cv=3,
        n_jobs=-1,
    )

    # Note: StackingClassifier's support for sample_weight is complex because it must
    # pass them to both the base estimators and the meta-estimator.
    # In scikit-learn >= 1.0, fit(X, y, sample_weight=sw) passes it to all base estimators.
    if sample_weights is not None:
        logger.info("  Applying time-decay sample weights to ensemble.")
        # Scikit-learn's StackingClassifier will route sample_weight down to the base models.
        stack_clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        stack_clf.fit(X_train, y_train)

    logger.info("Ensemble training complete. Calibrating probabilities...")

    # --- Calibrated wrapper ---
    # We wrap the entire ensemble in isotonic calibration.
    # cv="prefit" means we calibrate on the already trained ensemble using cross-validation internally,
    # OR we can just use cv=3 which will do 3-fold cv on the whole stack (very slow).
    # To save time but maintain calibration, we use cv=3, but it trains the stack 3 times.
    # For a production ensemble, it's safer to use cv=2 or cv=3.
    
    calibrated_clf = CalibratedClassifierCV(
        estimator=stack_clf,
        method="isotonic",
        cv=2,  # 2-fold CV to save compute time while ensuring calibration
    )

    calibrated_clf.fit(X_train, y_train, sample_weight=sample_weights)

    logger.info("Model calibration complete.")

    return calibrated_clf


def save_model(model: CalibratedClassifierCV, path: str) -> None:
    """Save the trained model to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to: %s", path)


def load_model(path: str) -> CalibratedClassifierCV:
    """Load a trained model from disk."""
    model = joblib.load(path)
    logger.info("Model loaded from: %s", path)
    return model
