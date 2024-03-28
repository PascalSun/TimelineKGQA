import os
import zipfile

import pandas as pd
from sqlalchemy import create_engine, text

from tkgqa_generator.constants import (
    DATA_ICEWS_DICTS_DATA_DIR,
    DATA_ICEWS_EVENTS_DATA_DIR,
    DB_CONNECTION_STR,
    DOC_DIR,
)
from tkgqa_generator.utils import get_logger
import plotly.graph_objects as go
import argparse

logger = get_logger(__name__)


class ICEWSDataLoader:
    def __init__(self, data_type="all", view_sector_tree_web: bool = False):
        self.engine = create_engine(DB_CONNECTION_STR)
        self.data_type = data_type
        self.view_sector_tree_web = view_sector_tree_web

    def icews_load_data(self):
        """
        Before doing anything, you will need to download the ICEWS data from the Harvard Dataverse.

        Data name is: "ICEWS Coded Event Data",
            https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075

        After downloading the data, extract the zip file and put the data in the following directory:
            tkgqa_generator/tkgqa_generator/data/icews_events_data/ICEWS/ICEWS Coded Event Data

        :return:
        """

        if self.data_type == "all" or self.data_type == "icews":

            # loop the folder, and unzip all the files ending with .zip
            # it will override the data if it is running twice
            for file in os.listdir(DATA_ICEWS_EVENTS_DATA_DIR):
                if file.endswith(".zip"):
                    # unzip the file
                    zip_path = DATA_ICEWS_EVENTS_DATA_DIR / file
                    logger.info(f"Unzipping {zip_path}")
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(DATA_ICEWS_EVENTS_DATA_DIR)

            # read all .tab or .csv files into the df and check their column distribution
            # pandas read from tab, csv files
            # have done the check, all files have the consistent same column names
            combined_df = None
            for file in os.listdir(DATA_ICEWS_EVENTS_DATA_DIR):
                if file.endswith(".tab") or file.endswith(".csv"):
                    tab_path = DATA_ICEWS_EVENTS_DATA_DIR / file
                    logger.info(f"Reading {tab_path}")
                    df = pd.read_csv(tab_path, sep="\t", low_memory=False)
                    if combined_df is None:
                        combined_df = df
                    else:
                        # combine df and combined_df
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
            logger.info("Loading data into database")
            combined_df.to_sql(
                "icews", con=self.engine, if_exists="replace", index=False
            )

        if self.data_type == "all" or self.data_type == "icews_dicts":
            # load the ICEWS Dictionaries into the database for further review
            logger.info("Loading dictionaries into database")
            # loop all the files in the directory, and saving them into the database
            for file in os.listdir(DATA_ICEWS_DICTS_DATA_DIR):
                if file.endswith(".csv"):
                    csv_path = DATA_ICEWS_DICTS_DATA_DIR / file
                    logger.info(f"Reading {csv_path}")
                    # if sector in the file name, then the csv do not have header
                    if "sector" in file:
                        df = pd.read_csv(csv_path, header=None, low_memory=False)
                    else:
                        df = pd.read_csv(csv_path, low_memory=False)
                    table_name = file.rsplit(".", 2)[0].replace(".", "_")
                    logger.info(f"Loading {table_name} into database")
                    df.to_sql(
                        table_name, con=self.engine, if_exists="replace", index=False
                    )

    def icews_explore_data(self):
        """
        Read the ICEWS_Sector, as it is a tree, plot a tree for this.
        :return:
        """

        df = pd.read_sql_table("icews_sectors", con=self.engine)
        # Initialize lists to hold the transformed data
        names = []
        parents = []

        logger.info(df.head())

        # Track the last seen name at each level to establish parent-child relationships
        last_seen = {-1: ""}  # Root has no name

        # Iterate over the rows in the original dataframe
        for _, row in df.iterrows():
            for level in range(len(row)):
                logger.info(f"Level: {level}")
                # Check if the cell is not empty
                if not pd.isnull(row[level]):
                    # This level's name
                    name = row[level]
                    # Parent is the last seen name in the previous level
                    parent = last_seen[level - 1]
                    # Update this level's last seen name
                    last_seen[level] = name
                    # If this name at this level is not already added, add it to the lists
                    if not name in names or parents[names.index(name)] != parent:
                        names.append(name)
                        parents.append(parent)
                    break  # Move to the next row once the first non-empty cell is processed

        # Creating a new dataframe from the transformed data
        transformed_df = pd.DataFrame({"name": names, "parent": parents})

        # Display the first few rows of the transformed dataframe
        logger.info(transformed_df.head())

        # Creating a tree diagram with Plotly
        fig = go.Figure(
            go.Treemap(
                labels=transformed_df["name"],
                parents=transformed_df["parent"],
            )
        )

        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

        if self.view_sector_tree_web:
            fig.show()

    def icews_actor_unified_kg(self):
        """
        run sql query
        ```sql
        CREATE TABLE unified_kg_icews_actor AS
        SELECT
            "Actor Name" AS subject,
            json_build_object('Country', "Country", 'Aliases', "Aliases") AS subject_json,
            'Affiliation To' AS predicate,
            '{}'::json AS predicate_json, -- Correctly cast empty JSON object
            "Affiliation To" AS object,
            '{}'::json AS object_json, -- Correctly cast empty JSON object
            "Affiliation Start Date" AS start_time,
            "Affiliation End Date" AS end_time
        FROM
            icews_actors;
        ```
        :return:
        """
        cursor = self.engine.connect()
        cursor.execute(
            text(
                """
            DO
            $$
                BEGIN
                    -- Attempt to create the table if it doesn't exist
                    -- This part only creates the table structure
                    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'unified_kg_icews_actor') THEN
                        CREATE TABLE public.unified_kg_icews_actor
                        (
                            subject        TEXT,
                            subject_json   JSON,
                            predicate      TEXT,
                            predicate_json JSON DEFAULT '{}'::json,
                            object         TEXT,
                            object_json    JSON DEFAULT '{}'::json,
                            start_time     DATE,
                            end_time       DATE
                        );
                    END IF;
                    TRUNCATE TABLE public.unified_kg_icews_actor;
                    INSERT INTO unified_kg_icews_actor(
                        subject,
                        subject_json,
                        predicate,
                        predicate_json,
                        object,
                        object_json,
                        start_time,
                        end_time
                    )
                    SELECT
                        "Actor Name" AS subject,
                        json_build_object('Country', "Country", 'Aliases', "Aliases") AS subject_json,
                        'Affiliation To' AS predicate,
                        '{}'::json AS predicate_json, -- Correctly cast empty JSON object
                        "Affiliation To" AS object,
                        '{}'::json AS object_json, -- Correctly cast empty JSON object
                        "Affiliation Start Date" AS start_time,
                        "Affiliation End Date" AS end_time
                    FROM
                        icews_actors;
                END
            $$;

            """
            )
        )
        cursor.commit()
        cursor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ICEWS data to DB")
    parser.add_argument(
        "--load_data",
        type=str,
        help="Which dataset to load into the database, icews or icews_dicts",
        default="none",
    )
    parser.add_argument(
        "--explore_data_view_sector",
        action="store_true",
        help="Whether to view the sector tree in a web browser",
        default=False,
    )
    args = parser.parse_args()
    icews_data_loader = ICEWSDataLoader(
        data_type=args.load_data, view_sector_tree_web=args.explore_data_view_sector
    )
    icews_data_loader.icews_load_data()
    icews_data_loader.icews_explore_data()
    icews_data_loader.icews_actor_unified_kg()
