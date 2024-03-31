import argparse
import json
import os
import time
import zipfile

import pandas as pd
import plotly.graph_objects as go
import torch
from sqlalchemy import create_engine, text
from transformers import BertModel, BertTokenizer

from tkgqa_generator.constants import (
    DATA_DIR,
    DATA_ICEWS_DICTS_DATA_DIR,
    DATA_ICEWS_EVENTS_DATA_DIR,
    DB_CONNECTION_STR,
    DOC_DIR,
)
from tkgqa_generator.utils import API, get_logger, timer

logger = get_logger(__name__)


class ICEWSDataLoader:
    def __init__(
        self,
        data_type="all",
        view_sector_tree_web: bool = False,
        token: str = "",
        queue_name: str = "",
    ):
        self.engine = create_engine(DB_CONNECTION_STR)
        self.data_type = data_type
        self.view_sector_tree_web = view_sector_tree_web
        self.api = API(token=token)
        self.queue_name = queue_name

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
                    if "id" not in df.columns:
                        df["id"] = range(1, 1 + len(df))
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

    def icews_actor_embedding(self, model_name: str = "Mixtral-8x7b"):
        """
        embedding iceews actors with several models, add columns to original table
        embedding content will be subject affiliated to object
        :return:
        """
        # add a json field for the embedding, then we can have {"model_name": "embedding"}
        add_embedding_column_sql = """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'icews_actors' AND column_name = 'embedding' AND table_schema = 'public'
            ) THEN
                ALTER TABLE public.icews_actors ADD COLUMN embedding JSONB;
            END IF;
        END
        $$;
        """
        cursor = self.engine.connect()
        cursor.execute(text(add_embedding_column_sql))
        cursor.commit()
        cursor.close()

        # loop the table, and get the embedding for each actor

        while True:
            # get the one that has not been embedded with SQL, embedding?.model_name is null
            with self.engine.connect() as conn:
                r = conn.execute(
                    text(
                        f"""
                        SELECT *
                        FROM icews_actors
                        WHERE id not in (SELECT id
                                         FROM icews_actors
                                         WHERE embedding ? '{model_name}')
                        ORDER BY id
                        DESC 
                        ;
                        """
                    )
                )
                result_no = 0
                for row in r.mappings():
                    logger.info(row)
                    result_no += 1
                    record_id = row["id"]
                    subject = row["Actor Name"]
                    object = row["Affiliation To"]
                    prompt = f"{subject} affiliated to {object}"
                    if model_name == "bert":
                        embedding = self.__icews_actor_bert_embedding(prompt)
                    else:
                        response = self.api.create_embedding(
                            prompt, model_name=model_name
                        )
                        embedding = response["data"][0]["embedding"]
                        # update the embedding column
                    with timer(
                        logger,
                        f"Updating embedding for {subject} affiliated to {object}",
                    ):
                        conn.execute(
                            text(
                                f"""
                                UPDATE icews_actors
                                SET embedding = jsonb_build_object('{model_name}', '{embedding}')
                                WHERE id = {record_id};
                                """
                            )
                        )
                        conn.commit()
                    time.sleep(0.3)

                if result_no == 0:
                    logger.info("All actors have been embedded")
                    break

    def icews_actor_queue_embedding(self, model_name: str = "Mixtral-8x7b"):
        """
        embedding iceews actors with several models, add columns to original table
        embedding content will be subject affiliated to object
        :return:
        """

        # get the one that has not been embedded with SQL, embedding?.model_name is null
        with self.engine.connect() as conn:
            r = conn.execute(
                text(
                    f"""
                        SELECT *
                        FROM icews_actors
                        WHERE id not in (SELECT id
                                         FROM icews_actors
                                         WHERE embedding ? '{model_name}')
                        ORDER BY id
                        DESC 
                        ;
                        """
                )
            )

            prompts = []
            for row in r.mappings():
                logger.info(row)
                record_id = row["id"]
                subject = row["Actor Name"]
                object = row["Affiliation To"]
                prompt = f"{subject} affiliated to {object}"
                prompts.append(prompt)

            # every 100 prompts, send to the queue
            for i in range(0, len(prompts), 100):
                if i + 100 > len(prompts):
                    response = self.api.queue_create_embedding(
                        prompts[i:], model_name=model_name, name=self.queue_name
                    )
                else:
                    response = self.api.queue_create_embedding(
                        prompts[i : i + 100],
                        model_name=model_name,
                        name=self.queue_name,
                    )
                time.sleep(0.3)

    def icews_actor_embedding_csv(self, queue_embedding_filename: str, model_name: str):
        conn = self.engine.connect()
        df = pd.read_csv(DATA_DIR / "ICEWS" / "processed" / queue_embedding_filename)
        for _, row in df.iterrows():
            prompt = row["prompt"]
            subject = prompt.split(" affiliated to ")[0].replace("'", "''")
            object = prompt.split(" affiliated to ")[1].replace("'", "''")
            embedding = json.loads(json.loads(row["response"]))["data"][0]["embedding"]

            # logger.debug(f"Subject: {subject}, Object: {object}, Embedding: {embedding}")

            # update the embedding column

            conn.execute(
                text(
                    f"""
                    UPDATE icews_actors
                    SET embedding = jsonb_build_object('{model_name}', '{embedding}')
                    WHERE "Actor Name" = '{subject}' AND "Affiliation To" = '{object}';
                    """
                )
            )
            conn.commit()

    def __icews_actor_bert_embedding(self, prompt: str):
        """
        Use the BERT model to embed the ICEWS actors
        :return:
        """
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Load pre-trained model
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()  # Set the model to evaluation mode
        encoded_input = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**encoded_input)
        last_hidden_states = outputs.last_hidden_state
        logger.info(last_hidden_states)
        return last_hidden_states

    def icews_actor_entity_resolution_check(self):
        """
        Check the entity resolution for the ICEWS actors
        :return:
        """
        pass

    def icews_actor_entity_timeline(self, actor_name: str):
        """
        SELECT * FROM icews_actors WHERE "Actor Name" = actor_name;
        :param actor_name:
        :return:
        """
        pass


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

    parser.add_argument(
        "--token",
        type=str,
        help="API token for the LLM model",
        default="f07d79cda0137abacd39258654090410650c2c0d",
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        help="Model name for the LLM model",
        default="Mixtral-8x7b",
    )

    parser.add_argument(
        "--queue_embedding_name",
        type=str,
        help="Queue embedding name",
        default=None,
    )
    parser.add_argument(
        "--queue_embedding_filename",
        type=str,
        help="Queue embedding filename",
        default=None,
    )
    args = parser.parse_args()
    icews_data_loader = ICEWSDataLoader(
        data_type=args.load_data,
        view_sector_tree_web=args.explore_data_view_sector,
        token=args.token,
        queue_name=args.queue_embedding_name,
    )
    icews_data_loader.icews_load_data()
    icews_data_loader.icews_explore_data()
    icews_data_loader.icews_actor_unified_kg()
    if args.queue_embedding_name:
        # this will cause timeout
        icews_data_loader.icews_actor_queue_embedding(model_name=args.llm_model_name)
    else:
        icews_data_loader.icews_actor_embedding(model_name=args.llm_model_name)
    # this is when finished the queue, and want to update the embedding
    if args.queue_embedding_filename:
        icews_data_loader.icews_actor_embedding_csv(
            queue_embedding_filename=args.queue_embedding_filename,
            model_name=args.llm_model_name,
        )
