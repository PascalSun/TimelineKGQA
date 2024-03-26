import os
import zipfile
from sqlalchemy import create_engine
from tkgqa_generator.utils import get_logger
import pandas as pd
from tkgqa_generator.constants import DATA_ICEWS_EVENTS_DATA_DIR, DB_CONNECTION_STR

logger = get_logger(__name__)


def load_icews_data():
    # loop the folder, and unzip all the files ending with .zip
    # it will override the data if it is running twice
    for file in os.listdir(DATA_ICEWS_EVENTS_DATA_DIR):
        if file.endswith(".zip"):
            # unzip the file
            zip_path = DATA_ICEWS_EVENTS_DATA_DIR / file
            logger.info(f"Unzipping {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_ICEWS_EVENTS_DATA_DIR)

    # read all .tab or .csv files into the df and check their column distribution
    # pandas read from tab, csv files
    # have done the check, all files have the consistent same column names
    combined_df = None
    for file in os.listdir(DATA_ICEWS_EVENTS_DATA_DIR):
        if file.endswith(".tab") or file.endswith(".csv"):
            tab_path = DATA_ICEWS_EVENTS_DATA_DIR / file
            logger.info(f"Reading {tab_path}")
            df = pd.read_csv(
                tab_path, sep="\t", low_memory=False
            )
            if combined_df is None:
                combined_df = df
            else:
                # combine df and combined_df
                combined_df = pd.concat(
                    [combined_df, df], ignore_index=True
                )
    engine = create_engine(DB_CONNECTION_STR)
    logger.info("Loading data into database")
    combined_df.to_sql(
        "icews", con=engine, if_exists="append", index=False

    )


if __name__ == '__main__':
    load_icews_data()
