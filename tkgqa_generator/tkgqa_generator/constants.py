import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"

DATA_ICEWS = "ICEWS"
DATA_ICEWS_EVENTS = "ICEWS Coded Event Data"

DATA_ICEWS_EVENTS_DATA_DIR = DATA_DIR / DATA_ICEWS / DATA_ICEWS_EVENTS
