import json

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sqlalchemy import create_engine, text
from tqdm import tqdm

from tkgqa_generator.utils import get_logger, timer

logger = get_logger(__name__)

client = OpenAI()


class Question2SQL:
    """
    Get the question and the table schema and return the SQL query
    """

    def __init__(
            self,
            table_name: str,
            host: str,
            port: int,
            user: str,
            password: str,
            db_name: str,
    ):
        """
        Args:
            table_name: The name of the table
            host: The host of the database
            port: The port of the database
            user: The user of the database
            password: The password of the database
            db_name: The name of the database



        """
        self.table_name = table_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name

        self.engine = create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        )

        # create a table if not exists to store the text2sql questions and results
        with self.engine.connect() as connection:
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name}_text2sql (
                        id SERIAL PRIMARY KEY,
                        question TEXT,
                        question_level TEXT,
                        prompt TEXT,
                        sql_query TEXT,
                        correct BOOLEAN
                    );
                    """
                )
            )
            connection.commit()

    def process_question_to_prompt(self, question: str, table_schema: str):
        """
        Process the question to the prompt

        Args:
            question: The question
            table_schema: The table schema

        Returns:
            The prompt
        """
        prompt = f"""question: {question}
        The related knowledge to answer this question is in table {table_schema},
        the table name is {self.table_name},
        table_schema:
        {table_schema}
        Generate the sql query to retrieve the relevant information from the table to answer the question.
        Return all columns for the rows that satisfy the condition.
        Return the SQL query  in json format with the key "sql_query"
        """
        return prompt

    def process_question_to_prompt_with_semantic_parse(self, question: str, table_schema: str):
        # TODO: and question mark, should we do this? and How?
        """
        Process the question to the prompt

        Args:
            question: The question
            table_schema: The table schema

        Returns:
            The prompt
        """
        prompt = f"""question: {question}
        The related knowledge to answer this question is in table {table_schema},
        the table name is {self.table_name},
        table_schema:
        {table_schema}
        Generate the sql query to retrieve the relevant information from the table to answer the question.
        Return all columns for the rows that satisfy the condition.
        Return the SQL query  in json format with the key "sql_query"
        """
        return prompt

    @staticmethod
    def text2sql_gpt(prompt: str, model_name: str = "gpt-3.5-turbo"):
        """
        Get the question and the table schema and return the SQL query

        Args:
            prompt(str): The prompt
            model_name: The model name
        """
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert about text to SQL in PostgreSQL database
                                      """,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            logger.debug(f"Response: {response.choices[0].message.content}")
            sql_query = response.choices[0].message.content
            query_json = json.loads(sql_query)
            sql_query = query_json["sql_query"]
            return sql_query
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return ""

    def benchmark(
            self,
            question_level: str = "complex",
            random_eval: bool = False,
    ):
        """

        Args:
            question_level: The level of the question
            random_eval: If True, evaluate the model with random questions
        """
        if random_eval:
            questions_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions ORDER BY RANDOM() LIMIT 100",
                self.engine,
            )
        elif question_level == "all":
            questions_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions", self.engine
            )

        else:
            questions_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE question_level = '{question_level}'",
                self.engine,
            )

        logger.info(f"Number of questions: {len(questions_df)}")

        table_schema = self.get_table_schema()
        logger.info(f"Table Schema: {table_schema}")

        prompt_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_text2sql",
            self.engine,
        )

        for index, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
            question = row["question"]
            if question in prompt_df["question"].values:
                continue
            prompt = self.process_question_to_prompt(question, table_schema)
            logger.info(f"Question: {question}")
            logger.info(f"Table Schema: {table_schema}")
            logger.info(f"Prompt: {prompt}")

            query = text(
                f"""
                          INSERT INTO {self.table_name}_text2sql (question, question_level, prompt)
                          VALUES (:question, :question_level, :prompt)
                      """
            )

            with self.engine.connect() as connection:
                connection.execute(
                    query,
                    {
                        "question": question,
                        "question_level": question_level,
                        "prompt": prompt,
                    },
                )
                connection.commit()

        prompt_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_text2sql",
            self.engine,
        )

        for index, row in tqdm(prompt_df.iterrows(), total=len(prompt_df)):
            question = row["question"]
            prompt = row["prompt"]
            if row["sql_query"] is not None:
                continue
            sql = self.text2sql_gpt(
                prompt=prompt,
                model_name="gpt-3.5-turbo",
            )
            logger.info(f"SQL Query: {sql}")
            # add the record to the table

            query = text(
                f"""
                          UPDATE {self.table_name}_text2sql
                          SET sql_query = :sql_query
                          WHERE question = :question
                      """
            )

            with self.engine.connect() as connection:
                connection.execute(
                    query,
                    {
                        "question": question,
                        "sql_query": sql,
                    },
                )
                connection.commit()

        # verify the result
        self.verify_results()

    def verify_results(self):
        prompts_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_text2sql WHERE sql_query IS NOT NULL",
            self.engine,
        )
        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions",
            self.engine,
        )

        for index, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
            question = row["question"]
            sql_query = row["sql_query"]
            if row["correct"] is not None:
                continue
            query = text(sql_query)
            try:
                df = pd.read_sql(query, self.engine)
                events = []
                for _, event in df.iterrows():
                    subject = event["subject"]
                    predicate = event["predicate"]
                    tail_object = event["object"]
                    start_time = event["start_time"]
                    end_time = event["end_time"]

                    events.append(
                        f"{subject}|{predicate}|{tail_object}|{start_time}|{end_time}"
                    )
                # locate the ground truth for the question
                ground_truth = questions_df[questions_df["question"] == question]
                ground_truth_events = ground_truth["events"].values.tolist()
                # decompose the nested list
                ground_truth_events = [
                    item for sublist in ground_truth_events for item in sublist
                ]
                logger.info(f"Question: {question}")
                logger.info(f"SQL Query: {sql_query}")
                logger.info(f"Events: {events}")
                logger.info(f"Ground Truth Events: {ground_truth_events}")
                if set(events) == set(ground_truth_events):
                    correct = True
                else:
                    correct = False
            except Exception as e:
                logger.exception(e)
                correct = False
            query = text(
                f"""
                          UPDATE {self.table_name}_text2sql
                          SET correct = :correct
                          WHERE question = :question
                      """
            )

            with self.engine.connect() as connection:
                connection.execute(
                    query,
                    {
                        "question": question,
                        "correct": correct,
                    },
                )
                connection.commit()

    def get_table_schema(self):
        query = f"""
        SELECT 
            column_name, 
            data_type, 
            character_maximum_length, 
            is_nullable, 
            column_default
        FROM 
            information_schema.columns
        WHERE 
            table_name = '{self.table_name}';
        """
        df = pd.read_sql(query, self.engine)
        return df.to_markdown(index=False)


if __name__ == "__main__":
    metric_question_level = "all"
    rag = Question2SQL(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )

    with timer(logger, "Benchmarking"):
        rag.benchmark(question_level=metric_question_level, random_eval=False)
