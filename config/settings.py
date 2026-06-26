"""
KickCast Configuration
======================
Loads all settings from environment variables (sourced from .env file).
Validates required values at import time — the app fails fast if anything
is missing rather than crashing later with a cryptic KeyError.

Design choice: We use a plain dataclass rather than a heavyweight config
library. Every setting has a sensible default where possible (Azurite
connection string, standard container names), but secrets like the API
key have no default and must be set explicitly.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (two levels up from this file: config/settings.py)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_path = _PROJECT_ROOT / ".env"

if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Also try .env.example as a fallback for structure reference,
    # but warn that real values are needed.
    _example_path = _PROJECT_ROOT / ".env.example"
    if _example_path.exists():
        load_dotenv(_example_path)
        print(
            "WARNING: No .env file found — loaded .env.example. "
            "Copy it to .env and fill in your real API key.",
            file=sys.stderr,
        )


def _require(var_name: str) -> str:
    """Return an env var's value, or crash with a clear message if missing."""
    value = os.getenv(var_name)
    if not value:
        print(
            f"FATAL: Required environment variable '{var_name}' is not set.\n"
            f"       Copy .env.example to .env and fill in the value.",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


def _get(var_name: str, default: str = "") -> str:
    """Return an env var's value, or a default."""
    return os.getenv(var_name, default)


# ---------------------------------------------------------------------------
# Azurite well-known connection string (used as default for local dev).
# This is Microsoft's published default — not a secret.
# ---------------------------------------------------------------------------
_AZURITE_CONN_STR = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq"
    "/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
    "TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
    "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings — one source of truth."""

    # Football API
    football_api_base_url: str
    football_api_key: str
    football_api_competition: str
    football_api_poll_interval_seconds: int

    # Historical data
    historical_data_path: str

    # Azure Storage
    azure_storage_connection_string: str

    # Blob container names
    blob_container_raw: str
    blob_container_processed: str

    # Table names
    table_fixtures: str
    table_predictions: str
    table_accuracy: str


def load_settings() -> Settings:
    """Build a Settings instance from the current environment."""
    return Settings(
        # API config — key is required, everything else has sensible defaults
        football_api_base_url=_get(
            "FOOTBALL_API_BASE_URL", "https://api.football-data.org/v4"
        ),
        football_api_key=_require("FOOTBALL_API_KEY"),
        football_api_competition=_get("FOOTBALL_API_COMPETITION", "WC"),
        football_api_poll_interval_seconds=int(
            _get("FOOTBALL_API_POLL_INTERVAL_SECONDS", "300")
        ),
        # Historical data path — defaults to project-relative location
        historical_data_path=_get(
            "HISTORICAL_DATA_PATH",
            str(_PROJECT_ROOT / "data" / "raw" / "results.csv"),
        ),
        # Azure Storage — defaults to Azurite
        azure_storage_connection_string=_get(
            "AZURE_STORAGE_CONNECTION_STRING", _AZURITE_CONN_STR
        ),
        # Container / table names
        blob_container_raw=_get("BLOB_CONTAINER_RAW", "kickcast-raw"),
        blob_container_processed=_get("BLOB_CONTAINER_PROCESSED", "kickcast-processed"),
        table_fixtures=_get("TABLE_FIXTURES", "fixtures"),
        table_predictions=_get("TABLE_PREDICTIONS", "predictions"),
        table_accuracy=_get("TABLE_ACCURACY", "accuracy"),
    )
