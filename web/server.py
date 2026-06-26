#!/usr/bin/env python3
"""
KickCast Web Dashboard Server
=============================
A lightweight Flask backend to serve the static frontend and provide
a JSON API fetching predictions from Azure Table Storage (Azurite).
"""

import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import load_settings
from data.storage import TableStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-server")

# Initialize Flask app
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # Enable CORS for frontend requests if needed

# Initialize settings and Table Store
try:
    settings = load_settings()
    table_store = TableStore(settings.azure_storage_connection_string)
    TABLE_NAME = "predictions"
except Exception as e:
    logger.error("Failed to initialize storage: %s", e)
    table_store = None
    TABLE_NAME = ""


@app.route("/")
def index():
    """Serve the main dashboard HTML."""
    return send_from_directory(".", "index.html")


@app.route("/api/matches")
def api_matches():
    """
    Fetch the latest predictions from Azure Table Storage and return as JSON.
    We sort by MatchDate to ensure chronological order.
    """
    if table_store is None:
        return jsonify({"error": "Storage not configured."}), 500

    try:
        # We query the predictions table. The partition key is usually 'wc2026'.
        entities = table_store.query_entities(TABLE_NAME)
        
        # Format for frontend
        matches = []
        for e in entities:
            matches.append({
                "id": e.get("RowKey"),
                "date": e.get("Date"),
                "stage": e.get("Stage"),
                "status": e.get("Status"),
                "actual_home_score": e.get("ActualHomeScore"),
                "actual_away_score": e.get("ActualAwayScore"),
                "home_pen_score": e.get("HomePenScore"),
                "away_pen_score": e.get("AwayPenScore"),
                "is_aet": e.get("IsAet", False),
                "home_scorers": e.get("HomeScorers"),
                "away_scorers": e.get("AwayScorers"),
                "home_team": e.get("HomeTeam"),
                "away_team": e.get("AwayTeam"),
                "home_elo": e.get("HomeElo"),
                "away_elo": e.get("AwayElo"),
                "prob_home": e.get("HomeWinProb"),
                "prob_draw": e.get("DrawProb"),
                "prob_away": e.get("AwayWinProb"),
                "prediction": e.get("PredictedOutcome"),
                "confidence": e.get("Confidence"),
                "pred_home_score": e.get("PredictedHomeScore"),
                "pred_away_score": e.get("PredictedAwayScore"),
                "advance_method": e.get("AdvanceMethod"),
                "aet_home_prob": e.get("AetHomeProb"),
                "aet_away_prob": e.get("AetAwayProb"),
                "pen_home_prob": e.get("PenHomeProb"),
                "pen_away_prob": e.get("PenAwayProb"),
                "updated_at": e.get("UpdatedAt")
            })

        # Sort chronologically
        matches.sort(key=lambda x: x["date"] if x["date"] else "9999")
        
        return jsonify({"matches": matches})

    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/match/<match_id>/summary', methods=['GET'])
def get_match_summary(match_id):
    """
    Proxy to ESPN's summary API to get live rosters, lineups, and key players.
    """
    import requests
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={match_id}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return jsonify(resp.json())
        return jsonify({"error": "ESPN API returned error"}), resp.status_code
    except Exception as e:
        logger.error(f"Error fetching match summary for {match_id}: {e}")
        return jsonify({"error": "Failed to fetch summary"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting KickCast Dashboard Server on port %d...", port)
    app.run(host="0.0.0.0", port=port, debug=True)
