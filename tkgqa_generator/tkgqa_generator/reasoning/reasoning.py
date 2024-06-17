import argparse
import random
from multiprocessing import cpu_count

import pandas as pd
import tqdm
from sqlalchemy import create_engine

from tkgqa_generator.reasoning.openai_reasoning import reasoning_temporal_questions
from tkgqa_generator.utils import get_logger, timer

logger = get_logger(__name__)


class TemporalReasoningEvaluator:
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
            table_name (str): The table name to be evaluated.
            host (str): The host of the database.
            port (int): The port of the database.
            user (str): The user of the database.
            password (str): The password of the database.
            db_name (str): The database name. Default is "tkgqa".
        """

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.engine = create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        )
        self.table_name = table_name
        with timer(logger, "Loading the table"):
            self.qa_df = pd.read_sql_table(table_name, self.engine)
            logger.info(
                f"Loaded {len(self.qa_df)} records from the table {table_name}."
            )

        self.max_workers = cpu_count

    def evaluate(self):
        """
        Evaluate the temporal reasoning. via the LLM
        Given the question, the LLM will generate the answer.
        """
        random_index = random.randint(0, len(self.qa_df))

        for index, row in tqdm.tqdm(self.qa_df.iterrows(), total=len(self.qa_df)):
            if index != random_index:
                continue
            question = row["question"]
            events = row["events"]
            logger.info(f"Question: {question}")
            logger.info(f"Events: {events}")
            logger.info(f"Answer: {row['answer']}")
            reasoning_prompt = (
                f"Given the events: {events}, reason the following question: {question}"
            )

            res = reasoning_temporal_questions(reasoning_prompt)

            logger.info(f"Reasoning result: {res}")
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--table_name", type=str, default="unified_kg_cron_questions")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--user", type=str, default="tkgqa")
    parser.add_argument("--password", type=str, default="tkgqa")
    parser.add_argument("--db_name", type=str, default="tkgqa")
    args = parser.parse_args()

    evaluator = TemporalReasoningEvaluator(
        table_name=args.table_name,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        db_name=args.db_name,
    )
    evaluator.evaluate()
