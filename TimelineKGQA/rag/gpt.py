import concurrent.futures
from multiprocessing import cpu_count
from typing import List

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text
from tqdm import tqdm

from TimelineKGQA.constants import LOGS_DIR
from TimelineKGQA.openai_utils import embedding_content
from TimelineKGQA.rag.metrics import hit_n, mean_reciprocal_rank
from TimelineKGQA.utils import get_logger, timer

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

            if "embedding" in self.event_df.columns:
                try:
                    self.event_df["embedding"] = self.event_df["embedding"].apply(
                        lambda x: list(map(float, x[1:-1].split(",")))
                    )
                except Exception as e:
                    logger.error(e)
        self.max_workers = cpu_count()

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
                    f"SELECT column_name FROM information_schema.columns WHERE "
                    f"table_name = '{self.table_name}_questions' "
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

    def _process_fact_row(self, row):
        """
        Args:
            row: The row of the fact

        """
        content = f"{row['subject']} {row['predicate']} {row['object']} {row['start_time']} {row['end_time']}"
        embedding = embedding_content(content)
        with self.engine.connect() as cursor:
            cursor.execute(
                text(
                    f"UPDATE {self.table_name} SET embedding = array{embedding}::vector "
                    f"WHERE id = {row['id']};"
                ),
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
            f"SELECT * FROM {self.table_name} WHERE embedding IS NULL;",
            self.engine,
        )
        # embed the facts
        # check df size, if it is empty, then return
        if df.shape[0] == 0:
            return
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = []
            for index, row in tqdm(
                df.iterrows(), total=df.shape[0], desc="Embedding Facts"
            ):
                futures.append(executor.submit(self._process_fact_row, row))
            concurrent.futures.wait(futures)

    def _process_question_row(self, row):
        embedding = embedding_content(row["question"])
        with self.engine.connect() as cursor:
            cursor.execute(
                text(
                    f"UPDATE {self.table_name}_questions SET embedding = array{embedding}::vector "
                    f"WHERE id = {row['id']};"
                ),
            )
            cursor.commit()

    def embed_questions(
        self, question_level: str = "complex", random_eval: bool = False
    ):
        """
        Get all the questions into the embedding, and save the embedding

        Args:
            question_level: The level of the question, can be complex, medium, simple
            random_eval (bool): Whether to do the random evaluation
        """
        # get whether the question with embedding total number = 2000, if yes, do not need to continue
        if random_eval:
            df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NULL ORDER BY RANDOM() LIMIT 2000;",
                self.engine,
            )
        elif question_level == "all":
            df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NULL;",
                self.engine,
            )
        else:
            df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NULL "
                f"and question_level = '{question_level}';",
                self.engine,
            )

        # embed the facts
        # check df size, if it is empty, then return
        if df.shape[0] == 0:
            return
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = []
            for index, row in tqdm(
                df.iterrows(), total=df.shape[0], desc="Embedding Questions"
            ):
                futures.append(executor.submit(self._process_question_row, row))
            concurrent.futures.wait(futures)

    def benchmark(
        self,
        question_level: str = "complex",
        random_eval: bool = False,
        semantic_parse: bool = False,
    ):
        """
        Benchmark the RAG model

        Need to first do the embedding of the questions

        2000 questions * 100000 facts = 2000000 comparisons

        For each question, select top 30 fact index.

        Grab the fact, and then compare with the actual fact.

        Args:
            question_level: The level of the question, can be complex, medium, simple
            random_eval (bool): Whether to do the random evaluation
            semantic_parse: Whether to do the semantic parse

        """

        if random_eval:
            questions_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL ORDER BY RANDOM() LIMIT 2000;",
                self.engine,
            )
        elif question_level == "all":
            questions_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL;",
                self.engine,
            )
        else:
            questions_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL "
                f"AND question_level = '{question_level}';",
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
        logger.info(similarities.shape)

        if semantic_parse:
            """
            Filter the facts based on the entity
            if it is candidate, mark as 1
            if it is not candidate, mark as 0
            Then use this to mask the similarity matrix
            not candidate one set as -inf
            Then do the top 30
            """
            mask_result_matrix = self.semantic_parse(questions_df)
            similarities = similarities + mask_result_matrix

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
            "medium": 2,
            "simple": 1,
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
        ranks_df.to_csv(LOGS_DIR / f"{question_level}_ranks.csv")
        mrr = mean_reciprocal_rank(ranks_df["rank"].tolist())
        logger.info(
            f"MRR: {mrr}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_1 = hit_n(ranks_df["rank"].tolist(), 1)
        logger.info(
            f"Hit@1: {hit_1}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_3 = hit_n(ranks_df["rank"].tolist(), 3)
        logger.info(
            f"Hit@3: {hit_3}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_5 = hit_n(ranks_df["rank"].tolist(), 5)
        logger.info(
            f"Hit@5: {hit_5}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_10 = hit_n(ranks_df["rank"].tolist(), 10)
        logger.info(
            f"Hit@10: {hit_10}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )

        # summary based on the question level
        mrr_simple = mean_reciprocal_rank(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1]
        )
        logger.info(
            f"MRR: {mrr_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_1_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 1
        )
        logger.info(
            f"Hit@1: {hit_1_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_3_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 3
        )
        logger.info(
            f"Hit@3: {hit_3_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_5_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 5
        )

        logger.info(
            f"Hit@5: {hit_5_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_10_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 10
        )

        logger.info(
            f"Hit@10: {hit_10_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        mrr_medium = mean_reciprocal_rank(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2]
        )
        logger.info(
            f"MRR: {mrr_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_1_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 1
        )

        logger.info(
            f"Hit@1: {hit_1_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_3_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 3
        )

        logger.info(
            f"Hit@3: {hit_3_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_5_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 5
        )

        logger.info(
            f"Hit@5: {hit_5_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_10_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 10
        )

        logger.info(
            f"Hit@10: {hit_10_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        mrr_complex = mean_reciprocal_rank(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3]
        )

        logger.info(
            f"MRR: {mrr_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_1_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 1
        )

        logger.info(
            f"Hit@1: {hit_1_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_3_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 3
        )

        logger.info(
            f"Hit@3: {hit_3_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_5_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 5
        )

        logger.info(
            f"Hit@5: {hit_5_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_10_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 10
        )

        logger.info(
            f"Hit@10: {hit_10_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

    def semantic_parse(self, questions_df: pd.DataFrame):
        """
        Filter the facts based on the entity
        if it is candidate, mark as 1
        if it is not candidate, mark as 0
        Then use this to mask the similarity matrix
        not candidate one set as -inf
        Then do the top 30

        Return will be a len(questions_df) x len(events_df) matrix, with 1 and 0
        """

        def extract_entities(events: List[str]):
            """
            Extract the entities from the event

            Args:
                events: The event string

            Returns:
                The entities in the event
            """
            the_entities = []
            for event in events:
                try:
                    elements = event.split("|")
                    the_entities.append(elements[0])
                    the_entities.append(elements[2])
                except Exception as e:
                    logger.debug(e)

            return the_entities

        questions_df["entities"] = questions_df["events"].apply(
            lambda x: extract_entities(x)
        )
        # get all value to be -2
        result_matrix = np.zeros(
            (len(questions_df), len(self.event_df)), dtype="float64"
        )
        result_matrix = result_matrix - 2
        for index, row in tqdm(
            questions_df.iterrows(), total=questions_df.shape[0], desc="Semantic Parse"
        ):
            entities = row["entities"]
            for entity in entities:
                result_matrix[index] = np.where(
                    self.event_df["subject"] == entity, 1, result_matrix[index]
                )
                result_matrix[index] = np.where(
                    self.event_df["object"] == entity, 1, result_matrix[index]
                )
        return result_matrix


if __name__ == "__main__":
    metric_question_level = "all"
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
        rag.embed_questions(question_level=metric_question_level, random_eval=False)

    with timer(logger, "Benchmark without semantic parse"):
        rag.benchmark(question_level=metric_question_level, random_eval=False)

    with timer(logger, "Benchmark with semantic parse"):
        rag.benchmark(
            question_level=metric_question_level, semantic_parse=True, random_eval=False
        )
