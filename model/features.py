"""
Feature Engineering Pipeline (v2)
==================================
Transforms the historical match DataFrame (with Elo columns) into a
feature matrix suitable for scikit-learn.

Feature catalog (16 features):
  1.  elo_home               — Home team's Elo rating before the match
  2.  elo_away               — Away team's Elo rating before the match
  3.  elo_diff               — elo_home - elo_away (net strength)
  4.  home_form              — Home team's points per match over last 5 matches
  5.  away_form              — Away team's points per match over last 5 matches
  6.  home_goals_avg         — Home team's avg goals scored over last 10 matches
  7.  away_goals_avg         — Away team's avg goals scored over last 10 matches
  8.  home_goals_conceded_avg— Home team's avg goals conceded over last 10
  9.  away_goals_conceded_avg— Away team's avg goals conceded over last 10
  10. is_neutral             — Whether the match is at a neutral venue (0/1)
  11. h2h_home_win_rate      — Home team's historical win rate vs this opponent
  12. h2h_goal_diff_avg      — Home team's avg goal diff vs this opponent
  13. is_knockout            — Whether the match is a knockout/elimination game
  14. attack_vs_defense_home — home_goals_avg - away_goals_conceded_avg
  15. attack_vs_defense_away — away_goals_avg - home_goals_conceded_avg
  16. form_diff              — home_form - away_form

All features are computed from the data, never hardcoded.
Rolling-window features (form, goals) are computed per-team,
chronologically, using only matches played before the current one.

Target variable:
  result: H=0 (home win), D=1 (draw), A=2 (away win)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names (used consistently for training and prediction)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "home_form",
    "away_form",
    "home_goals_avg",
    "away_goals_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "is_neutral",
    # v2 features
    "h2h_home_win_rate",
    "h2h_goal_diff_avg",
    "is_knockout",
    "attack_vs_defense_home",
    "attack_vs_defense_away",
    "form_diff",
    # v3 features
    "home_win_streak",
    "away_win_streak",
    "home_clean_sheet_pct",
    "away_clean_sheet_pct",
    "home_goals_std",
    "away_goals_std",
]

# Target encoding: must match between training and prediction
RESULT_MAP = {"H": 0, "D": 1, "A": 2}
RESULT_LABELS = {0: "H", 1: "D", 2: "A"}

# Rolling window sizes
FORM_WINDOW = 5   # Last 5 matches for form
GOALS_WINDOW = 10  # Last 10 matches for goals average
H2H_WINDOW = 10   # Last 10 head-to-head meetings

# Knockout stage identifiers (matched case-insensitively against tournament/stage)
KNOCKOUT_KEYWORDS = [
    "round of 32", "round of 16", "quarter", "semi", "final",
    "3rd place", "third place", "knockout", "elimination",
]


def _is_knockout_stage(tournament: str, stage: str = "") -> bool:
    """Determine if a match is in a knockout/elimination stage."""
    text = f"{tournament} {stage}".lower()
    return any(kw in text for kw in KNOCKOUT_KEYWORDS)


def _compute_team_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team rolling statistics: form, goals average, and H2H.

    This iterates chronologically and, for each match, looks up the
    most recent N matches for each team to compute:
      - form: points per match (W=3, D=1, L=0) over last FORM_WINDOW
      - goals_avg: average goals scored over last GOALS_WINDOW
      - goals_conceded_avg: average goals conceded over last GOALS_WINDOW
      - h2h_home_win_rate: home team's historical win rate against opponent
      - h2h_goal_diff_avg: home team's avg goal diff against opponent

    The values are point-in-time: only matches played BEFORE the current
    one are used.
    """
    # Per-team match history: list of (points_earned, goals_scored, goals_conceded, did_win)
    team_histories: dict[str, list[tuple[float, int, int, bool]]] = {}
    # Per-matchup history: key=(teamA, teamB), value=list of (goal_diff_for_A, win_for_A)
    h2h_histories: dict[tuple[str, str], list[tuple[int, float]]] = {}

    home_form_list = []
    away_form_list = []
    home_goals_list = []
    away_goals_list = []
    home_goals_conceded_list = []
    away_goals_conceded_list = []
    h2h_home_wr_list = []
    h2h_gd_list = []
    # v3 lists
    home_win_streak_list = []
    away_win_streak_list = []
    home_cs_pct_list = []
    away_cs_pct_list = []
    home_goals_std_list = []
    away_goals_std_list = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        # --- Compute features BEFORE updating history (point-in-time) ---

        # Home team form
        home_hist = team_histories.get(home_team, [])
        if len(home_hist) >= FORM_WINDOW:
            recent = home_hist[-FORM_WINDOW:]
            home_form_list.append(
                sum(pts for pts, _, _, _ in recent) / FORM_WINDOW
            )
        else:
            # Not enough history — use average or neutral value
            home_form_list.append(
                sum(pts for pts, _, _, _ in home_hist) / len(home_hist)
                if home_hist else 1.0  # neutral default: 1 point/match
            )

        # Away team form
        away_hist = team_histories.get(away_team, [])
        if len(away_hist) >= FORM_WINDOW:
            recent = away_hist[-FORM_WINDOW:]
            away_form_list.append(
                sum(pts for pts, _, _, _ in recent) / FORM_WINDOW
            )
        else:
            away_form_list.append(
                sum(pts for pts, _, _, _ in away_hist) / len(away_hist)
                if away_hist else 1.0
            )

        # Home team goals average
        if len(home_hist) >= GOALS_WINDOW:
            recent = home_hist[-GOALS_WINDOW:]
            home_goals_list.append(
                sum(g for _, g, _, _ in recent) / GOALS_WINDOW
            )
            home_goals_conceded_list.append(
                sum(gc for _, _, gc, _ in recent) / GOALS_WINDOW
            )
        else:
            home_goals_list.append(
                sum(g for _, g, _, _ in home_hist) / len(home_hist)
                if home_hist else 1.0  # neutral default
            )
            home_goals_conceded_list.append(
                sum(gc for _, _, gc, _ in home_hist) / len(home_hist)
                if home_hist else 1.0  # neutral default
            )

        # Away team goals average
        if len(away_hist) >= GOALS_WINDOW:
            recent = away_hist[-GOALS_WINDOW:]
            away_goals_list.append(
                sum(g for _, g, _, _ in recent) / GOALS_WINDOW
            )
            away_goals_conceded_list.append(
                sum(gc for _, _, gc, _ in recent) / GOALS_WINDOW
            )
        else:
            away_goals_list.append(
                sum(g for _, g, _, _ in away_hist) / len(away_hist)
                if away_hist else 1.0
            )
            away_goals_conceded_list.append(
                sum(gc for _, _, gc, _ in away_hist) / len(away_hist)
                if away_hist else 1.0
            )

        # Head-to-head: home team's record against away team
        # Look at both orderings: (home, away) and (away, home)
        h2h_records = []
        for key, flip in [((home_team, away_team), False), ((away_team, home_team), True)]:
            entries = h2h_histories.get(key, [])
            for gd, win in entries:
                if flip:
                    h2h_records.append((-gd, 1.0 - win if win != 0.5 else 0.5))
                else:
                    h2h_records.append((gd, win))

        if h2h_records:
            recent_h2h = h2h_records[-H2H_WINDOW:]
            h2h_home_wr_list.append(
                sum(w for _, w in recent_h2h) / len(recent_h2h)
            )
            h2h_gd_list.append(
                sum(gd for gd, _ in recent_h2h) / len(recent_h2h)
            )
        else:
            h2h_home_wr_list.append(0.5)  # neutral: no history
            h2h_gd_list.append(0.0)

        # --- v3: Win streak, clean sheet %, goal std dev ---
        def _compute_streak_and_stats(hist):
            """Compute win streak, clean sheet %, and goals std dev from history."""
            if not hist:
                return 0, 0.0, 0.5
            # Win streak: count consecutive wins from the end
            streak = 0
            for pts, _, _, won in reversed(hist[-GOALS_WINDOW:]):
                if won:
                    streak += 1
                else:
                    break
            # Clean sheet %: % of recent matches where team conceded 0
            recent = hist[-GOALS_WINDOW:]
            clean_sheets = sum(1 for _, _, gc, _ in recent if gc == 0)
            cs_pct = clean_sheets / len(recent)
            # Goals std dev: consistency of scoring
            goals = [g for _, g, _, _ in recent]
            goals_std = float(np.std(goals)) if len(goals) > 1 else 0.5
            return streak, cs_pct, goals_std

        h_streak, h_cs, h_gstd = _compute_streak_and_stats(home_hist)
        a_streak, a_cs, a_gstd = _compute_streak_and_stats(away_hist)
        home_win_streak_list.append(h_streak)
        away_win_streak_list.append(a_streak)
        home_cs_pct_list.append(h_cs)
        away_cs_pct_list.append(a_cs)
        home_goals_std_list.append(h_gstd)
        away_goals_std_list.append(a_gstd)

        # --- Update history AFTER computing features ---

        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        # Home team: points earned and goals scored
        if home_score > away_score:
            home_pts, away_pts = 3.0, 0.0
            home_win, away_win = 1.0, 0.0
            h_won, a_won = True, False
        elif home_score < away_score:
            home_pts, away_pts = 0.0, 3.0
            home_win, away_win = 0.0, 1.0
            h_won, a_won = False, True
        else:
            home_pts, away_pts = 1.0, 1.0
            home_win, away_win = 0.5, 0.5
            h_won, a_won = False, False

        team_histories.setdefault(home_team, []).append((home_pts, home_score, away_score, h_won))
        team_histories.setdefault(away_team, []).append((away_pts, away_score, home_score, a_won))

        # H2H history: stored from home team's perspective
        goal_diff = home_score - away_score
        h2h_histories.setdefault((home_team, away_team), []).append((goal_diff, home_win))

    df = df.copy()
    df["home_form"] = home_form_list
    df["away_form"] = away_form_list
    df["home_goals_avg"] = home_goals_list
    df["away_goals_avg"] = away_goals_list
    df["home_goals_conceded_avg"] = home_goals_conceded_list
    df["away_goals_conceded_avg"] = away_goals_conceded_list
    df["h2h_home_win_rate"] = h2h_home_wr_list
    df["h2h_goal_diff_avg"] = h2h_gd_list
    df["home_win_streak"] = home_win_streak_list
    df["away_win_streak"] = away_win_streak_list
    df["home_clean_sheet_pct"] = home_cs_pct_list
    df["away_clean_sheet_pct"] = away_cs_pct_list
    df["home_goals_std"] = home_goals_std_list
    df["away_goals_std"] = away_goals_std_list

    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build the full feature matrix and target vector from the historical
    DataFrame (which must already have Elo columns from elo.py).

    Args:
        df: Historical DataFrame with columns:
            elo_home, elo_away, elo_diff, home_score, away_score,
            home_team, away_team, result, neutral, tournament, date

    Returns:
        (df_featured, X, y) where:
          df_featured: the DataFrame with all feature columns added
          X: numpy array of shape (n_matches, 16) — the feature matrix
          y: numpy array of shape (n_matches,) — target (0=H, 1=D, 2=A)
    """
    logger.info("Building features for %d matches...", len(df))

    # Ensure the DataFrame has Elo columns
    required_elo_cols = {"elo_home", "elo_away", "elo_diff"}
    if not required_elo_cols.issubset(set(df.columns)):
        raise ValueError(
            f"DataFrame missing Elo columns: {required_elo_cols - set(df.columns)}. "
            "Run EloRatingSystem.compute_all_ratings() first."
        )

    # Compute rolling team stats (form, goals, H2H)
    df = _compute_team_rolling_stats(df)

    # Neutral venue flag (boolean → int)
    df["is_neutral"] = df["neutral"].astype(int)

    # Tournament stage feature: is it a knockout match?
    stage_col = df.get("stage", pd.Series([""] * len(df)))
    tournament_col = df["tournament"]
    df["is_knockout"] = [
        1.0 if _is_knockout_stage(t, s) else 0.0
        for t, s in zip(tournament_col, stage_col if "stage" in df.columns else [""] * len(df))
    ]

    # Cross-features: attack strength vs opposing defense weakness
    df["attack_vs_defense_home"] = df["home_goals_avg"] - df["away_goals_conceded_avg"]
    df["attack_vs_defense_away"] = df["away_goals_avg"] - df["home_goals_conceded_avg"]

    # Form difference
    df["form_diff"] = df["home_form"] - df["away_form"]

    # Build feature matrix
    X = df[FEATURE_COLUMNS].values.astype(np.float64)

    # Build target vector
    y = df["result"].map(RESULT_MAP).values.astype(np.int64)

    # Log feature statistics
    logger.info("Feature matrix shape: %s", X.shape)
    for i, col in enumerate(FEATURE_COLUMNS):
        logger.info(
            "  %s: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
            col,
            np.mean(X[:, i]),
            np.std(X[:, i]),
            np.min(X[:, i]),
            np.max(X[:, i]),
        )

    logger.info(
        "Target distribution: H=%d (%.1f%%), D=%d (%.1f%%), A=%d (%.1f%%)",
        np.sum(y == 0), 100 * np.mean(y == 0),
        np.sum(y == 1), 100 * np.mean(y == 1),
        np.sum(y == 2), 100 * np.mean(y == 2),
    )

    return df, X, y


def compute_features_for_fixture(
    home_team: str,
    away_team: str,
    is_neutral: bool,
    elo_ratings: dict[str, float],
    match_history: pd.DataFrame,
    stage: str = "Group Stage",
) -> np.ndarray:
    """
    Compute features for a single upcoming fixture.

    This is the prediction-time feature computation. It uses the
    current Elo ratings and recent match history (from the historical
    dataset plus any recent results).

    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        is_neutral: Whether the venue is neutral
        elo_ratings: Current Elo ratings dict {team_name: elo}
        match_history: DataFrame of recent matches (for form/goals)
        stage: Tournament stage name (e.g., "Group Stage", "Quarterfinals")

    Returns:
        numpy array of shape (1, 16) — one row of features
    """
    default_elo = 1500.0

    elo_home = elo_ratings.get(home_team, default_elo)
    elo_away = elo_ratings.get(away_team, default_elo)
    elo_diff = elo_home - elo_away

    # Compute recent form for each team from match history
    home_form = _team_recent_form(home_team, match_history, FORM_WINDOW)
    away_form = _team_recent_form(away_team, match_history, FORM_WINDOW)

    # Compute recent goals average
    home_goals_avg, home_goals_conceded_avg = _team_recent_goals(home_team, match_history, GOALS_WINDOW)
    away_goals_avg, away_goals_conceded_avg = _team_recent_goals(away_team, match_history, GOALS_WINDOW)

    # H2H features
    h2h_wr, h2h_gd = _team_h2h_record(home_team, away_team, match_history, H2H_WINDOW)

    # Knockout flag
    is_knockout = 1.0 if _is_knockout_stage("FIFA World Cup", stage) else 0.0

    # Cross features
    attack_vs_defense_home = home_goals_avg - away_goals_conceded_avg
    attack_vs_defense_away = away_goals_avg - home_goals_conceded_avg
    form_diff = home_form - away_form
    
    # v3 features
    h_streak, h_cs, h_gstd = _team_streak_and_stats(home_team, match_history, GOALS_WINDOW)
    a_streak, a_cs, a_gstd = _team_streak_and_stats(away_team, match_history, GOALS_WINDOW)

    features = np.array([[
        elo_home,
        elo_away,
        elo_diff,
        home_form,
        away_form,
        home_goals_avg,
        away_goals_avg,
        home_goals_conceded_avg,
        away_goals_conceded_avg,
        1.0 if is_neutral else 0.0,
        h2h_wr,
        h2h_gd,
        is_knockout,
        attack_vs_defense_home,
        attack_vs_defense_away,
        form_diff,
        h_streak,
        a_streak,
        h_cs,
        a_cs,
        h_gstd,
        a_gstd,
    ]])

    return features


def _team_recent_form(
    team: str, df: pd.DataFrame, window: int
) -> float:
    """
    Compute a team's recent form (points per match) from match history.

    Looks at the team's last `window` matches as either home or away,
    computes points earned (W=3, D=1, L=0), and returns the average.
    """
    # Find matches where this team played (as home or away)
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    team_matches = df[mask].tail(window)

    if len(team_matches) == 0:
        return 1.0  # neutral default

    points = []
    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            if row["home_score"] > row["away_score"]:
                points.append(3.0)
            elif row["home_score"] == row["away_score"]:
                points.append(1.0)
            else:
                points.append(0.0)
        else:
            if row["away_score"] > row["home_score"]:
                points.append(3.0)
            elif row["away_score"] == row["home_score"]:
                points.append(1.0)
            else:
                points.append(0.0)

    return sum(points) / len(points)


def _team_recent_goals(
    team: str, df: pd.DataFrame, window: int
) -> tuple[float, float]:
    """
    Compute a team's average goals scored and conceded over their last `window` matches.
    Returns (goals_scored_avg, goals_conceded_avg).
    """
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    team_matches = df[mask].tail(window)

    if len(team_matches) == 0:
        return 1.0, 1.0  # neutral default

    goals = []
    conceded = []
    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            goals.append(int(row["home_score"]))
            conceded.append(int(row["away_score"]))
        else:
            goals.append(int(row["away_score"]))
            conceded.append(int(row["home_score"]))

    return sum(goals) / len(goals), sum(conceded) / len(conceded)


def _team_h2h_record(
    home_team: str, away_team: str, df: pd.DataFrame, window: int
) -> tuple[float, float]:
    """
    Compute the home team's head-to-head record against the away team.

    Returns (win_rate, avg_goal_diff) from the home team's perspective,
    using the last `window` meetings between these two teams.
    """
    mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    h2h_matches = df[mask].tail(window)

    if len(h2h_matches) == 0:
        return 0.5, 0.0  # neutral default

    wins = 0.0
    goal_diffs = []
    for _, row in h2h_matches.iterrows():
        if row["home_team"] == home_team:
            gd = int(row["home_score"]) - int(row["away_score"])
            if gd > 0:
                wins += 1.0
            elif gd == 0:
                wins += 0.5
        else:
            gd = int(row["away_score"]) - int(row["home_score"])
            if gd > 0:
                wins += 1.0
            elif gd == 0:
                wins += 0.5
        goal_diffs.append(gd)

    return wins / len(h2h_matches), sum(goal_diffs) / len(goal_diffs)


def _team_streak_and_stats(
    team: str, df: pd.DataFrame, window: int
) -> tuple[int, float, float]:
    """
    Compute win streak, clean sheet %, and goal std dev for a single team.
    """
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    team_matches = df[mask].tail(window)
    
    if len(team_matches) == 0:
        return 0, 0.0, 0.5
        
    streak = 0
    clean_sheets = 0
    goals = []
    
    for _, row in reversed(list(team_matches.iterrows())):
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        
        if row["home_team"] == team:
            g = home_score
            gc = away_score
            won = home_score > away_score
        else:
            g = away_score
            gc = home_score
            won = away_score > home_score
            
        if won:
            streak += 1
        else:
            break
            
    for _, row in team_matches.iterrows():
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        
        if row["home_team"] == team:
            g = home_score
            gc = away_score
        else:
            g = away_score
            gc = home_score
            
        if gc == 0:
            clean_sheets += 1
        goals.append(g)
        
    cs_pct = clean_sheets / len(team_matches)
    goals_std = float(np.std(goals)) if len(goals) > 1 else 0.5
    
    return streak, cs_pct, goals_std
