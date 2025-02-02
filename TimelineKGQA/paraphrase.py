"""
Read a table of generated questions, and paraphrase them using OpenAI's GPT-4o model.

Given a table name, then read from the database, update the paraphrased questions, and write them back to the database.
"""

import argparse

import pandas as pd
import psycopg2
from loguru import logger
from tqdm import tqdm

from TimelineKGQA.openai_utils import (
    paraphrase_medium_question,
    paraphrase_simple_question,
)


class Paraphraser:
    def __init__(
        self,
        table_name: str,
        host: str = "localhost",
        port: int = 5433,
        database: str = "tkgqa",
        user: str = "tkgqa",
        password: str = "tkgqa",
    ):
        self.table_name = table_name
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

        self.connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        self.cursor = self.connection.cursor()

    def run(self):
        # Read questions from the database
        questions = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE paraphrased_question IS NULL",
            con=self.connection,
        )

        # Paraphrase questions using OpenAI's GPT-4o model
        for i, row in tqdm(questions.iterrows(), total=len(questions)):
            question = row["question"]
            level = row["question_level"]
            if level == "simple":
                paraphrased_question = paraphrase_simple_question(question)
            elif level == "medium":
                paraphrased_question = paraphrase_medium_question(question)
            elif level == "hard":
                paraphrased_question = paraphrase_medium_question(question)
            # update the paraphrased question in the database
            self.cursor.execute(
                f"UPDATE {self.table_name} SET paraphrased_question = %s WHERE id = %s",
                (paraphrased_question, row["id"]),
            )
            self.connection.commit()
            logger.info(
                f"Paraphrased question {row['id']}: {question} -> {paraphrased_question}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase questions using OpenAI's GPT-4o model."
    )
    parser.add_argument(
        "table_name", type=str, help="Name of the table containing the questions."
    )
    args = parser.parse_args()

    paraphraser = Paraphraser(args.table_name)
    paraphraser.run()


if __name__ == "__main__":
    main()
