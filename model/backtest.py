"""
Backtest Evaluation
===================
Evaluates the trained model against a held-out test set using a strict
chronological split (no look-ahead bias).

Reports three metrics for both the model and a naive baseline:
  1. Accuracy — % of correct class predictions
  2. Log Loss — penalizes confident wrong predictions (lower = better)
  3. Brier Score — mean squared error of probabilities (lower = better)

The naive baseline is "always predict the higher Elo team wins":
  - If elo_home > elo_away: predict Home Win
  - If elo_home < elo_away: predict Away Win
  - If equal: predict Draw
  - For probabilities: use the Elo expected-score formula directly

All metrics are MEASURED from real data — never invented or hardcoded.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

from model.elo import expected_result
from model.features import RESULT_MAP, RESULT_LABELS

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Container for measured backtest metrics."""
    name: str
    accuracy: float
    log_loss_score: float
    brier_score: float
    n_test_samples: int

    # Per-class accuracy
    home_accuracy: float
    draw_accuracy: float
    away_accuracy: float

    def __str__(self) -> str:
        return (
            f"\n{'=' * 50}\n"
            f"  {self.name}\n"
            f"  Test samples: {self.n_test_samples}\n"
            f"{'=' * 50}\n"
            f"  Accuracy:    {self.accuracy:.4f} ({self.accuracy * 100:.1f}%)\n"
            f"  Log Loss:    {self.log_loss_score:.4f}\n"
            f"  Brier Score: {self.brier_score:.4f}\n"
            f"{'─' * 50}\n"
            f"  Per-class accuracy:\n"
            f"    Home Win:  {self.home_accuracy:.4f} ({self.home_accuracy * 100:.1f}%)\n"
            f"    Draw:      {self.draw_accuracy:.4f} ({self.draw_accuracy * 100:.1f}%)\n"
            f"    Away Win:  {self.away_accuracy:.4f} ({self.away_accuracy * 100:.1f}%)\n"
        )


def compute_brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the multi-class Brier score.

    Brier score = mean over all samples of sum over classes of
    (predicted_prob - actual_indicator)^2

    Lower is better. Perfect = 0.0. Random guessing ≈ 0.67 for 3 classes.
    """
    n_classes = y_prob.shape[1]
    n_samples = len(y_true)

    # One-hot encode the true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1.0

    # Brier score: mean squared difference
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))


def evaluate_model(
    model: CalibratedClassifierCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "Model",
) -> BacktestResults:
    """
    Evaluate a trained model on the test set.

    Args:
        model: Trained scikit-learn classifier with predict_proba()
        X_test: Test feature matrix
        y_test: Test target vector (0=H, 1=D, 2=A)
        name: Label for the results

    Returns:
        BacktestResults with measured metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Ensure probabilities cover all 3 classes
    if y_prob.shape[1] < 3:
        logger.warning(
            "Model only predicts %d classes (expected 3). "
            "This may happen if the test set is small.",
            y_prob.shape[1],
        )

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob, labels=[0, 1, 2])
    brier = compute_brier_score_multiclass(y_test, y_prob)

    # Per-class accuracy
    home_mask = y_test == 0
    draw_mask = y_test == 1
    away_mask = y_test == 2

    home_acc = accuracy_score(y_test[home_mask], y_pred[home_mask]) if home_mask.sum() > 0 else 0.0
    draw_acc = accuracy_score(y_test[draw_mask], y_pred[draw_mask]) if draw_mask.sum() > 0 else 0.0
    away_acc = accuracy_score(y_test[away_mask], y_pred[away_mask]) if away_mask.sum() > 0 else 0.0

    return BacktestResults(
        name=name,
        accuracy=acc,
        log_loss_score=ll,
        brier_score=brier,
        n_test_samples=len(y_test),
        home_accuracy=home_acc,
        draw_accuracy=draw_acc,
        away_accuracy=away_acc,
    )


def evaluate_baseline(
    X_test: np.ndarray,
    y_test: np.ndarray,
    elo_home_idx: int = 0,
    elo_away_idx: int = 1,
) -> BacktestResults:
    """
    Evaluate the naive baseline: "always predict the higher-Elo team wins."

    For class prediction:
      - If elo_home > elo_away → predict Home Win (0)
      - If elo_home < elo_away → predict Away Win (2)
      - If equal → predict Draw (1)

    For probability estimates:
      - Use the Elo expected-score formula to derive probabilities
      - Home win prob = expected_result(elo_home, elo_away)
      - Away win prob = 1 - home_win_prob
      - Draw prob = allocated from the "uncertain middle"

    Args:
        X_test: Test feature matrix
        y_test: Test target vector
        elo_home_idx: Column index for elo_home in X
        elo_away_idx: Column index for elo_away in X

    Returns:
        BacktestResults with measured baseline metrics
    """
    n_samples = len(y_test)
    y_pred = np.zeros(n_samples, dtype=np.int64)
    y_prob = np.zeros((n_samples, 3), dtype=np.float64)

    for i in range(n_samples):
        elo_home = X_test[i, elo_home_idx]
        elo_away = X_test[i, elo_away_idx]

        # Expected result from Elo formula
        home_expected = expected_result(elo_home, elo_away)
        away_expected = 1.0 - home_expected

        # Derive probabilities:
        # The Elo expected result gives P(home wins or draws) roughly.
        # We split it into 3 classes using a simple scheme:
        #   - If the teams are close (expected ≈ 0.5), draw is more likely
        #   - The "draw probability" is peaked when teams are evenly matched
        draw_prob = max(0.0, 1.0 - 3.0 * abs(home_expected - 0.5))
        draw_prob = min(draw_prob, 0.35)  # cap draw probability

        # Distribute remaining probability between home and away
        remaining = 1.0 - draw_prob
        home_prob = remaining * home_expected
        away_prob = remaining * away_expected

        y_prob[i] = [home_prob, draw_prob, away_prob]

        # Class prediction
        if elo_home > elo_away:
            y_pred[i] = 0  # Home win
        elif elo_home < elo_away:
            y_pred[i] = 2  # Away win
        else:
            y_pred[i] = 1  # Draw

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob, labels=[0, 1, 2])
    brier = compute_brier_score_multiclass(y_test, y_prob)

    # Per-class accuracy
    home_mask = y_test == 0
    draw_mask = y_test == 1
    away_mask = y_test == 2

    home_acc = accuracy_score(y_test[home_mask], y_pred[home_mask]) if home_mask.sum() > 0 else 0.0
    draw_acc = accuracy_score(y_test[draw_mask], y_pred[draw_mask]) if draw_mask.sum() > 0 else 0.0
    away_acc = accuracy_score(y_test[away_mask], y_pred[away_mask]) if away_mask.sum() > 0 else 0.0

    return BacktestResults(
        name="Baseline (Higher Elo Wins)",
        accuracy=acc,
        log_loss_score=ll,
        brier_score=brier,
        n_test_samples=len(y_test),
        home_accuracy=home_acc,
        draw_accuracy=draw_acc,
        away_accuracy=away_acc,
    )


def print_comparison(model_results: BacktestResults, baseline_results: BacktestResults) -> None:
    """Print a side-by-side comparison of model vs baseline."""
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS — Model vs. Baseline")
    print("=" * 60)
    print(f"  Test samples: {model_results.n_test_samples}")
    print()
    print(f"  {'Metric':<20s} {'Model':>12s} {'Baseline':>12s} {'Δ':>10s}")
    print(f"  {'─' * 54}")

    # Accuracy (higher is better)
    acc_delta = model_results.accuracy - baseline_results.accuracy
    acc_better = "✓" if acc_delta > 0 else ""
    print(
        f"  {'Accuracy':<20s} {model_results.accuracy:>11.4f} "
        f"{baseline_results.accuracy:>12.4f} {acc_delta:>+9.4f} {acc_better}"
    )

    # Log Loss (lower is better)
    ll_delta = model_results.log_loss_score - baseline_results.log_loss_score
    ll_better = "✓" if ll_delta < 0 else ""
    print(
        f"  {'Log Loss':<20s} {model_results.log_loss_score:>11.4f} "
        f"{baseline_results.log_loss_score:>12.4f} {ll_delta:>+9.4f} {ll_better}"
    )

    # Brier Score (lower is better)
    bs_delta = model_results.brier_score - baseline_results.brier_score
    bs_better = "✓" if bs_delta < 0 else ""
    print(
        f"  {'Brier Score':<20s} {model_results.brier_score:>11.4f} "
        f"{baseline_results.brier_score:>12.4f} {bs_delta:>+9.4f} {bs_better}"
    )

    print()
    print(f"  {'Per-class accuracy:'}")
    print(
        f"  {'  Home Win':<20s} {model_results.home_accuracy:>11.4f} "
        f"{baseline_results.home_accuracy:>12.4f}"
    )
    print(
        f"  {'  Draw':<20s} {model_results.draw_accuracy:>11.4f} "
        f"{baseline_results.draw_accuracy:>12.4f}"
    )
    print(
        f"  {'  Away Win':<20s} {model_results.away_accuracy:>11.4f} "
        f"{baseline_results.away_accuracy:>12.4f}"
    )
    print()
