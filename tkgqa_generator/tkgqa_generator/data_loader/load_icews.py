import argparse
import json
import os
import time
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sentence_transformers import SentenceTransformer
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
                    logger.debug(row)
                    result_no += 1
                    record_id = row["id"]
                    subject = row["Actor Name"]
                    object = row["Affiliation To"]
                    prompt = f"{subject} affiliated to {object}"
                    logger.info(f"Embedding {prompt}")
                    if model_name == "bert":
                        embedding = self.__icews_actor_bert_embedding(prompt)
                    else:
                        raise ValueError(f"Model name {model_name} not supported")
                    with timer(
                        logger,
                        f"Updating embedding for {subject} affiliated to {object}",
                    ):
                        conn.execute(
                            text(
                                f"""
                                UPDATE icews_actors
                                SET embedding = jsonb_set(
                                                coalesce(embedding, '{{}}')::jsonb, 
                                                array['{model_name}'], 
                                                '"{embedding}"'::jsonb
                                            )
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
            logger.info(self.queue_name)
            logger.info(model_name)
            for row in r.mappings():
                logger.debug(row)
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
            if row["model_name"] != model_name:
                continue
            prompt = row["prompt"]
            subject = prompt.split(" affiliated to ")[0].replace("'", "''")
            object = prompt.split(" affiliated to ")[1].replace("'", "''")
            if model_name == "bert":
                embedding = json.loads(json.loads(row["response"]))["embedding"]
            else:
                embedding = json.loads(json.loads(row["response"]))["data"][0][
                    "embedding"
                ]
            # logger.debug(f"Subject: {subject}, Object: {object}, Embedding: {embedding}")

            # update the embedding column
            conn.execute(
                text(
                    f"""
                    UPDATE icews_actors
                    SET embedding = jsonb_set(
                                        coalesce(embedding, '{{}}'::jsonb), 
                                        array['{model_name}'], 
                                        '"{embedding}"'::jsonb
                                    )
                    WHERE "Actor Name" = '{subject}' AND "Affiliation To" = '{object}';
                    """
                )
            )
            logger.debug(
                f"""WHERE "Actor Name" = '{subject}' AND "Affiliation To" = '{object}'"""
            )
            conn.commit()

    def __icews_actor_bert_embedding(self, prompt: str):
        """
        Use the BERT model to embed the ICEWS actors
        :return:
        """
        # Load pre-trained model tokenizer (vocabulary)
        # Generate embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(prompt)
        return embeddings.tolist()

    @staticmethod
    def __similarity_to_color(similarity):
        # Assuming similarity ranges from -1 to 1, normalize to 0-1
        # normalized_similarity = (similarity + 1) / 2
        # Use a colormap (e.g., 'RdYlGn' for Red-Yellow-Green)
        return plt.get_cmap("RdYlGn")(similarity)

    def icews_actor_entity_resolution_check(self):
        """
        Check the entity resolution for the ICEWS actors
        :return:
        """
        pass

    def icews_actor_subject_count_distribution(self, actor_name: str):
        """
        Get all records for the actor_name and present the occurrence across a timeline.
        X-axis: Year
        Y-axis: Month
        When hovering over a point, it shows the value of "Affiliation To".
        """
        # SQL query to get all records for the specified actor_name
        get_all_records_for_actor_name = f"""
        SELECT 
        "Actor Name",
        "Affiliation Start Date",
        "Affiliation End Date",
        "Affiliation To",
        embedding
        FROM icews_actors WHERE "Actor Name" = '{actor_name}';
        """
        # Execute the query
        actor_df = pd.read_sql_query(get_all_records_for_actor_name, con=self.engine)

        # Replace placeholders with extreme dates for ease of handling
        actor_df["Affiliation Start Date"] = actor_df["Affiliation Start Date"].replace(
            "beginning of time", "1990-01-01"
        )
        actor_df["Affiliation End Date"] = actor_df["Affiliation End Date"].replace(
            "end of time", "2015-12-31"
        )

        # Convert dates to datetime format
        actor_df["Affiliation Start Date"] = pd.to_datetime(
            actor_df["Affiliation Start Date"]
        )
        actor_df["Affiliation End Date"] = pd.to_datetime(
            actor_df["Affiliation End Date"]
        )

        # Extract year and month for both start and end dates
        actor_df["start_year"] = actor_df["Affiliation Start Date"].dt.year
        actor_df["start_month"] = actor_df["Affiliation Start Date"].dt.month
        actor_df["end_year"] = actor_df["Affiliation End Date"].dt.year
        actor_df["end_month"] = actor_df["Affiliation End Date"].dt.month

        # order by start year and month
        actor_df = actor_df.sort_values(by=["start_year", "start_month"])
        actor_df = actor_df.reset_index(drop=True)

        # Prepare a figure object
        fig = go.Figure()

        # get the bert embedding for the first record
        embedding = actor_df.loc[0, "embedding"]

        first_embedding_value = embedding["bert"]
        # convert to tensor
        logger.info(f"First embedding: {type(first_embedding_value)}")
        first_embedding_value = torch.tensor(json.loads(first_embedding_value))
        # Iterate over each record to plot it
        for index, row in actor_df.iterrows():
            # Adding a line for each affiliation duration
            logger.info(row["start_year"])
            logger.info(index)
            embedding_value = row["embedding"]["bert"]
            embedding_value = torch.tensor(json.loads(embedding_value))
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(first_embedding_value),
                torch.tensor(embedding_value),
                dim=0,
            )
            logger.info(f"Similarity: {similarity}")
            line_color = self.__similarity_to_color(similarity)
            fig.add_trace(
                go.Scatter(
                    x=[
                        row["start_year"] + row["start_month"] / 12,
                        row["end_year"] + row["end_month"] / 12,
                    ],
                    y=[index + 1, index + 1],
                    mode="lines+markers",
                    line=dict(color="rgb" + str(line_color[:3]), width=4),
                    name=row["Affiliation To"],
                    hoverinfo="text",
                    text=f"Affiliation To: {row['Affiliation To']}<br>Start: {row['start_year']}-{row['start_month']}<br>End: {row['end_year']}-{row['end_month']} <br>Similarity: {similarity:.2f}",
                )
            )

        # Update layout for readability
        fig.update_layout(
            title=f"Affiliation Timeline for {actor_name}",
            xaxis_title="Year",
            yaxis_title="index",
            yaxis=dict(
                tickmode="array",
                tickvals=actor_df.index.tolist(),
                ticktext=actor_df.index.tolist(),
            ),
        )

        fig.show()

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
        default=None,
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
    # parse the arguments
    args = parser.parse_args()
    # initialize the ICEWSDataLoader
    icews_data_loader = ICEWSDataLoader(
        data_type=args.load_data,
        view_sector_tree_web=args.explore_data_view_sector,
        token=args.token,
        queue_name=args.queue_embedding_name,
    )

    # load the data
    icews_data_loader.icews_load_data()
    # explore the data
    icews_data_loader.icews_explore_data()

    # create unified knowledge graph
    icews_data_loader.icews_actor_unified_kg()

    # create embeddings
    if args.queue_embedding_name and args.queue_embedding_name != "Individual":
        # this will cause timeout
        logger.info(f"Queue embedding: {args.queue_embedding_name}")
        icews_data_loader.icews_actor_queue_embedding(model_name=args.llm_model_name)
    elif args.queue_embedding_name == "Individual":
        logger.info(f"Individual embedding: {args.llm_model_name}")
        icews_data_loader.icews_actor_embedding(model_name=args.llm_model_name)
    else:
        logger.info("No need to create embeddings specified")
    # this is when finished the queue, and want to update the embedding
    if args.queue_embedding_filename and args.llm_model_name:
        logger.info(f"Process embedding filename: {args.queue_embedding_filename}")
        icews_data_loader.icews_actor_embedding_csv(
            queue_embedding_filename=args.queue_embedding_filename,
            model_name=args.llm_model_name,
        )

    # plot the distribution of the actor
    icews_data_loader.icews_actor_subject_count_distribution("Barack Obama")
