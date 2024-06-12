import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text
from tqdm import tqdm

from tkgqa_generator.openai_utils import embedding_content
from tkgqa_generator.rag.metrics import mean_reciprocal_rank
from tkgqa_generator.utils import get_logger, timer

logger = get_logger(__name__)


class RAGRank:
    """
    Input will be a list of the questions, and the output will be ranked facts based on the questions.

    """

    def __init__(
        self,
        table_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
        db_name: str = "tkgqa",
    ):
        """

        Args:
            table_name: The table name in the database, where the facts are stored
            host: The host of the database
            port: The port of the database
            user: The user of the database
            password: The password of the database
            db_name: The name of the database

        """
        # set up the db connection
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        )

        self.table_name = table_name
        with timer(logger, "Load Event Data"):
            self.event_df = pd.read_sql(
                f"SELECT * FROM {self.table_name};", self.engine
            )
            self.event_df["embedding"] = self.event_df["embedding"].apply(
                lambda x: list(map(float, x[1:-1].split(",")))
            )

    def add_embedding_column(self):
        """
        Add a column to the unified KG table, which will be the embedding of the fact.

        If the embedding column already exists, then we will not add the column.
        """
        # it is the sqlalchemy connection
        with self.engine.connect() as cursor:
            result = cursor.execute(
                text(
                    f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.table_name}' "
                    f"AND column_name = 'embedding';"
                )
            )
            if not result.fetchone():
                cursor.execute(
                    text(f"ALTER TABLE {self.table_name} ADD COLUMN embedding vector;")
                )

            cursor.commit()

        with self.engine.connect() as cursor:
            result = cursor.execute(
                text(
                    f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.table_name}_questions' "
                    f"AND column_name = 'embedding';"
                )
            )
            if not result.fetchone():
                cursor.execute(
                    text(
                        f"ALTER TABLE {self.table_name}_questions ADD COLUMN embedding vector;"
                    )
                )

                cursor.commit()

    def embed_facts(self):
        """
        Get all the facts into the embedding, and save the embedding


        Facts will be in unifed KG format, so we go into that table, and then grab the facts, and then embed them.

        We need to add a column to the unified KG table, which will be the embedding of the fact.
        """
        # get from embedding is None into dataframe
        df = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE embedding IS NULL;", self.engine
        )
        # embed the facts
        # check df size, if it is empty, then return
        if df.shape[0] == 0:
            return
        for index, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Embedding Facts"
        ):
            content = f"{row['subject']} {row['predicate']} {row['object']} {row['start_time']} {row['end_time']}"
            # logger.info(content)
            embedding = embedding_content(content)
            # logger.info(len(embedding))
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self.table_name} SET embedding = array{embedding}::vector WHERE id = {row['id']};",
                )
                self.connection.commit()

    def embed_questions(self):
        """
        Get all the questions into the embedding, and save the embedding

        """
        # get whether the question with embedding total number = 2000, if yes, do not need to continue

        df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL and question_level = 'complex';",
            self.engine,
        )

        if df.shape[0] >= 2000:
            return

        df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NULL and question_level = 'complex';",
            self.engine,
        )
        # embed the facts
        # check df size, if it is empty, then return
        if df.shape[0] == 0:
            return
        for index, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Embedding Questions"
        ):

            # logger.info(content)
            embedding = embedding_content(row["question"])
            # logger.info(len(embedding))
            with self.engine.connect() as cursor:
                cursor.execute(
                    text(
                        f"UPDATE {self.table_name}_questions SET embedding = array{embedding}::vector "
                        f"WHERE id = {row['id']};"
                    ),
                )
                cursor.commit()
            if index > 2000:
                return

    def benchmark(self):
        """
        Benchmark the RAG model

        Need to first do the embedding of the questions

        2000 questions * 100000 facts = 2000000 comparisons

        For each question, select top 30 fact index.

        Grab the fact, and then compare with the actual fact.

        """
        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL and question_level = 'complex' LIMIT 2000;",
            self.engine,
        )

        questions_df["embedding"] = questions_df["embedding"].apply(
            lambda x: list(map(float, x[1:-1].split(",")))
        )

        questions_embedding_array = np.array(
            questions_df["embedding"].tolist(), dtype="float64"
        )

        events_embedding_array = np.array(
            self.event_df["embedding"].tolist(), dtype="float64"
        )

        # calculate the similarity between the questions and the facts
        # 2000 questions x 100000 events
        # 2000 x 100000

        logger.info(questions_embedding_array.shape)
        logger.info(events_embedding_array.shape)
        similarities = torch.mm(
            torch.tensor(questions_embedding_array),
            torch.tensor(events_embedding_array).T,
        )
        # get top 30 event index based on the similarity
        # within the similarity matrix,
        top_30_values, top_30_indices = torch.topk(similarities, 30, dim=1)

        # loop the questions, get the top 30 events, compare with the actual events
        self.event_df["fact"] = (
            self.event_df["subject"]
            + "|"
            + self.event_df["predicate"]
            + "|"
            + self.event_df["object"]
            + "|"
            + self.event_df["start_time"]
            + "|"
            + self.event_df["end_time"]
        )

        ranks = []
        labels = {
            "complex": 3,
            "simple": 1,
            "medium": 2,
        }
        for index, row in tqdm(
            questions_df.iterrows(), total=questions_df.shape[0], desc="Benchmark"
        ):

            top_30_events = top_30_indices[index].tolist()
            # get all the events from self.events_df
            facts = self.event_df.iloc[top_30_events]["fact"].tolist()
            ids = self.event_df.iloc[top_30_events]["id"].tolist()
            relevant_facts = row["events"]

            rank = [1 if fact in relevant_facts else 0 for fact in facts]
            ranks.append(
                {
                    "question": row["question"],
                    "rank": {
                        "rank": rank,
                        "labels": labels[row["question_level"]],
                    },
                    "top_30_events": ids,
                }
            )
        ranks_df = pd.DataFrame(ranks)
        ranks_df.to_csv("ranks.csv")
        mrr = mean_reciprocal_rank(ranks_df["rank"].tolist())
        logger.info(f"MRR: {mrr}")


if __name__ == "__main__":
    rag = RAGRank(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    with timer(logger, "Add Embedding Column"):
        rag.add_embedding_column()

    with timer(logger, "Embed Facts"):
        rag.embed_facts()

    with timer(logger, "Embed Questions"):
        rag.embed_questions()

    with timer(logger, "Benchmark"):
        rag.benchmark()
