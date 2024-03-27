import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
DOC_DIR = PROJECT_DIR / "docs"

DATA_ICEWS = "ICEWS"
DATA_ICEWS_EVENTS = "ICEWS Coded Event Data"
DATA_ICEWS_DICTS = "ICEWS Dictionaries"

DATA_ICEWS_EVENTS_DATA_DIR = DATA_DIR / DATA_ICEWS / DATA_ICEWS_EVENTS
DATA_ICEWS_DICTS_DATA_DIR = DATA_DIR / DATA_ICEWS / DATA_ICEWS_DICTS

DB_NAME = "tkgqa"
DB_USER = "tkgqa"
DB_PASS = "tkgqa"
DB_HOST = "localhost"  # or the IP address of your database server
DB_PORT = "5433"
DB_CONNECTION_STR = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
