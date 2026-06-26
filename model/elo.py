"""
Elo Rating Engine
=================
Computes Elo ratings for every international football team from the full
match history. Each team starts at 1500 and their rating is updated
after every match based on the result vs. the expected result.

This implements a variant of the FIFA SUM algorithm:
  new_rating = old_rating + K * (actual_result - expected_result)

Where:
  - expected_result = 1 / (1 + 10^(-rating_diff / 600))
  - K depends on match importance (tournament type)
  - actual_result: 1.0 (win), 0.5 (draw), 0.0 (loss)

The K-factor is determined from the tournament name in the dataset,
not from any hardcoded team-level information.

Design choices:
  - We use 600 as the Elo divisor (matching FIFA's adaptation) rather
    than the chess-standard 400. This produces a wider spread that
    better reflects the variance in international football.
  - K-factors follow the FIFA importance tiers: World Cup matches
    move ratings more than friendlies.
  - All Elo values are point-in-time: each match row gets the Elo
    ratings as they stood *before* that match was played.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default starting Elo for all teams
# ---------------------------------------------------------------------------
DEFAULT_ELO = 1500.0

# ---------------------------------------------------------------------------
# Elo divisor — FIFA uses 600 (wider spread than chess's 400)
# ---------------------------------------------------------------------------
ELO_DIVISOR = 600.0

# ---------------------------------------------------------------------------
# K-factor tiers: determined by tournament importance.
# These patterns are matched against the `tournament` column in the
# historical dataset. The patterns are derived from actual tournament
# names in the data (e.g., "FIFA World Cup", "Friendly", etc.).
#
# Higher K means the match has more impact on ratings.
# ---------------------------------------------------------------------------
K_FACTOR_TIERS = [
    # (pattern_list, K_value, description)
    (
        ["FIFA World Cup"],
        50,
        "World Cup finals — highest stakes",
    ),
    (
        [
            "UEFA Euro",
            "Copa América",
            "African Cup of Nations",
            "AFC Asian Cup",
            "CONCACAF Gold Cup",
            "Confederations Cup",
            "UEFA Nations League",
        ],
        40,
        "Continental championship finals",
    ),
    (
        [
            "qualification",
            "Qualifying",
            "qualifiers",
        ],
        30,
        "World Cup / continental qualifiers",
    ),
    (
        ["Friendly"],
        20,
        "Friendly matches — lowest stakes",
    ),
]

# Default K for tournaments not matching any pattern
DEFAULT_K = 25


def _get_k_factor(tournament: str) -> int:
    """
    Determine the K-factor from the tournament name.

    Matches are checked against known patterns (case-insensitive).
    The first matching tier wins. This is pattern-based on real
    tournament names in the dataset, not hardcoded per team.
    """
    tournament_lower = tournament.lower()
    for patterns, k_value, _ in K_FACTOR_TIERS:
        for pattern in patterns:
            if pattern.lower() in tournament_lower:
                return k_value
    return DEFAULT_K


def expected_result(rating_a: float, rating_b: float) -> float:
    """
    Compute the expected match result for team A against team B.

    Returns a value between 0 and 1:
      ~1.0 = team A is overwhelmingly favored
      ~0.5 = evenly matched
      ~0.0 = team B is overwhelmingly favored

    Uses the FIFA variant with divisor 600.
    """
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / ELO_DIVISOR))


def _actual_result(home_score: int, away_score: int) -> tuple[float, float]:
    """
    Convert a match score to actual results for home and away teams.

    Returns (home_actual, away_actual):
      Win:  1.0 / 0.0
      Draw: 0.5 / 0.5
      Loss: 0.0 / 1.0
    """
    if home_score > away_score:
        return 1.0, 0.0
    elif home_score < away_score:
        return 0.0, 1.0
    else:
        return 0.5, 0.5


class EloRatingSystem:
    """
    Maintains and updates Elo ratings for all teams.

    Usage:
        elo = EloRatingSystem()
        df_with_elo = elo.compute_all_ratings(df)
        current_ratings = elo.get_ratings()
    """

    def __init__(self, default_elo: float = DEFAULT_ELO):
        self._default_elo = default_elo
        self._ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Get a team's current Elo rating (default if unseen)."""
        return self._ratings.get(team, self._default_elo)

    def get_ratings(self) -> dict[str, float]:
        """Get a copy of all current ratings."""
        return dict(self._ratings)

    def update(
        self, home_team: str, away_team: str,
        home_score: int, away_score: int,
        tournament: str,
    ) -> tuple[float, float]:
        """
        Update ratings for both teams after a match.

        Returns the (new_home_elo, new_away_elo) after the update.
        """
        # Get current ratings (or defaults for new teams)
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Compute expected results
        home_expected = expected_result(home_elo, away_elo)
        away_expected = 1.0 - home_expected

        # Compute actual results from scores
        home_actual, away_actual = _actual_result(home_score, away_score)

        # Get K-factor from tournament type
        k = _get_k_factor(tournament)

        # Update ratings
        new_home_elo = home_elo + k * (home_actual - home_expected)
        new_away_elo = away_elo + k * (away_actual - away_expected)

        self._ratings[home_team] = new_home_elo
        self._ratings[away_team] = new_away_elo

        return new_home_elo, new_away_elo

    def compute_all_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches chronologically and add Elo columns.

        IMPORTANT: The Elo values attached to each row are the ratings
        BEFORE the match was played (point-in-time). This prevents
        data leakage — we never use future information.

        Args:
            df: Historical match DataFrame, sorted by date, with columns:
                home_team, away_team, home_score, away_score, tournament

        Returns:
            The same DataFrame with added columns:
              - elo_home: home team's Elo before this match
              - elo_away: away team's Elo before this match
              - elo_diff: elo_home - elo_away
        """
        # Reset ratings
        self._ratings = {}

        elo_home_list = []
        elo_away_list = []

        for _, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            # Record PRE-MATCH Elo (point-in-time)
            elo_home_list.append(self.get_rating(home_team))
            elo_away_list.append(self.get_rating(away_team))

            # Update ratings with this match's result
            self.update(
                home_team=home_team,
                away_team=away_team,
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                tournament=row["tournament"],
            )

        df = df.copy()
        df["elo_home"] = elo_home_list
        df["elo_away"] = elo_away_list
        df["elo_diff"] = df["elo_home"] - df["elo_away"]

        # Log summary of final ratings
        sorted_teams = sorted(
            self._ratings.items(), key=lambda x: x[1], reverse=True
        )
        logger.info("Elo ratings computed for %d teams", len(self._ratings))
        logger.info("Top 10 teams by Elo:")
        for team, elo in sorted_teams[:10]:
            logger.info("  %4d  %s", int(elo), team)
        logger.info("Bottom 5 teams by Elo:")
        for team, elo in sorted_teams[-5:]:
            logger.info("  %4d  %s", int(elo), team)

        return df
