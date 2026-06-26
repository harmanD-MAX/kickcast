"""
ESPN Public API Client
======================
Fetches live World Cup 2026 fixtures and results from ESPN's free public JSON API.
This requires NO API key and is highly reliable for major tournaments.

No hardcoding of match data is used here. We parse real events from ESPN.
"""

import logging
import time
from typing import Any, Optional
import requests

from data.storage import BlobStore

logger = logging.getLogger(__name__)

STATUS_SCHEDULED = "SCHEDULED"
STATUS_TIMED = "TIMED"
STATUS_IN_PLAY = "IN_PLAY"
STATUS_FINISHED = "FINISHED"

class FootballAPIClient:
    MAX_RETRIES = 3
    BACKOFF_BASE_SECONDS = 1.0

    def __init__(
        self,
        base_url: str,
        api_key: str,  # Kept for backward compatibility but ignored
        competition: str, # Ignored, hardcoded via base_url for ESPN
        blob_store: BlobStore,
        cache_container: str,
    ):
        # We override the base URL to use the ESPN scoreboard endpoint directly
        self._base_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
        self._blob_store = blob_store
        self._cache_container = cache_container
        
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "KickCast/1.0"
        })

    def fetch_matches(self) -> Optional[dict]:
        return self._fetch_with_cache("?dates=20260601-20260720&limit=200", cache_key="cache/matches.json")

    def fetch_standings(self) -> Optional[dict]:
        return None  # ESPN handles standings differently, skipping for now

    def fetch_teams(self) -> Optional[dict]:
        return None

    def _fetch_with_cache(self, endpoint: str, cache_key: str) -> Optional[dict]:
        data = self._fetch_with_retry(endpoint)

        if data is not None:
            try:
                self._blob_store.upload_json(self._cache_container, cache_key, data)
            except Exception as e:
                logger.warning("Failed to cache API response: %s", e)
            return data

        logger.warning("API fetch failed — falling back to cached data")
        return self._blob_store.download_json(self._cache_container, cache_key)

    def _fetch_with_retry(self, endpoint: str) -> Optional[dict]:
        url = f"{self._base_url}{endpoint}"
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info("ESPN API request [attempt %d/%d]: GET %s", attempt, self.MAX_RETRIES, url)
                response = self._session.get(url, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code >= 500:
                    wait = self.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    time.sleep(wait)
                    continue
                    
                logger.error("Client error (%d)", response.status_code)
                return None
            except requests.RequestException as e:
                wait = self.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait)

        return None


def extract_matches(api_response: dict) -> list[dict]:
    """
    Transform ESPN's event structure into our flat format.
    """
    events = api_response.get("events", [])
    results = []

    for ev in events:
        comps = ev.get("competitions", [])
        if not comps: continue
        comp = comps[0]
        
        match_id = str(ev.get("id", ""))
        utc_date = ev.get("date", "")
        
        # Parse Status
        status_name = comp.get("status", {}).get("type", {}).get("name", "")
        period = comp.get("status", {}).get("period", 0)
        is_aet = ("AET" in status_name) or (period >= 4)
        
        # Determine status
        status = STATUS_SCHEDULED
        if "IN_PROGRESS" in status_name or "HALFTIME" in status_name:
            status = STATUS_IN_PLAY
        elif "FULL_TIME" in status_name or "FINAL" in status_name:
            status = STATUS_FINISHED
        else:
            status = status_name
            
        # Parse Teams
        home_team = ""
        away_team = ""
        home_scorers = ""
        away_scorers = ""
        
        home_team_id = ""
        away_team_id = ""
        home_pen_score = None
        away_pen_score = None
        
        competitors = comp.get("competitors", [])
        for team in competitors:
            name = team.get("team", {}).get("displayName", "")
            team_id = team.get("team", {}).get("id", "")
            score = team.get("score")
            shootout_score = team.get("shootoutScore")
            try:
                score_int = int(score) if score else None
                pen_score_int = int(shootout_score) if shootout_score else None
            except ValueError:
                score_int = None
                pen_score_int = None

            if team.get("homeAway") == "home":
                home_team = name
                home_score = score_int
                home_pen_score = pen_score_int
                home_team_id = team_id
            else:
                away_team = name
                away_score = score_int
                away_pen_score = pen_score_int
                away_team_id = team_id

        home_scorers_list = []
        away_scorers_list = []
        
        details = comp.get("details", [])
        for d in details:
            if d.get("scoringPlay"):
                team_id = d.get("team", {}).get("id")
                athletes = d.get("athletesInvolved", [])
                scorer_name = "Unknown"
                if athletes:
                    scorer_name = athletes[0].get("shortName", "")
                
                if d.get("penaltyKick"):
                    scorer_name += " (P)"
                elif d.get("ownGoal"):
                    scorer_name += " (OG)"

                if team_id == home_team_id:
                    home_scorers_list.append(scorer_name)
                elif team_id == away_team_id:
                    away_scorers_list.append(scorer_name)

        def format_scorers(scorers_list):
            counts = {}
            for s in scorers_list:
                counts[s] = counts.get(s, 0) + 1
            formatted = []
            for s, c in counts.items():
                if c > 1:
                    formatted.append(f"{s} (x{c})")
                else:
                    formatted.append(s)
            return ", ".join(formatted)

        home_scorers = format_scorers(home_scorers_list)
        away_scorers = format_scorers(away_scorers_list)

        # Parse Stage
        stage_slug = ev.get("season", {}).get("slug", "")
        stage_map = {
            "group-stage": "Group Stage",
            "round-of-32": "Round of 32",
            "round-of-16": "Round of 16",
            "quarterfinals": "Quarterfinals",
            "semifinals": "Semifinals",
            "3rd-place-match": "3rd Place Match",
            "final": "Final"
        }
        stage = stage_map.get(stage_slug, "World Cup")

        results.append({
            "match_id": match_id,
            "utc_date": utc_date,
            "status": status,
            "matchday": "",
            "stage": stage,
            "group": "",
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_pen_score": home_pen_score,
            "away_pen_score": away_pen_score,
            "is_aet": is_aet,
            "home_scorers": home_scorers,
            "away_scorers": away_scorers,
        })

    return results

def match_to_table_entity(match: dict) -> dict:
    return {
        "PartitionKey": "wc2026",
        "RowKey": match["match_id"],
        "utcDate": match["utc_date"],
        "status": match["status"],
        "matchday": match["matchday"],
        "stage": match["stage"],
        "group": match["group"],
        "homeTeam": match["home_team"],
        "awayTeam": match["away_team"],
        "homeScore": match["home_score"] if match["home_score"] is not None else -1,
        "awayScore": match["away_score"] if match["away_score"] is not None else -1,
        "homeScorers": match.get("home_scorers", ""),
        "awayScorers": match.get("away_scorers", ""),
    }
