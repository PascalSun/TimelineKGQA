import concurrent.futures
import json
from multiprocessing import cpu_count
from typing import List

import backoff
import pandas as pd
from openai import OpenAI, RateLimitError
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
                        prompt_semantic TEXT,
                        sql_query TEXT,
                        sql_query_semantic TEXT,
                        correct BOOLEAN,
                        correct_semantic BOOLEAN
                    );
                    """
                )
            )
            connection.commit()

    def benchmark(
        self,
        semantic_parse: bool = False,
    ):
        """

        Args:
            semantic_parse: If True, use the semantic parse to generate the prompt
        """
        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions",
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

            events = row["events"]
            prompt_semantic = self.process_question_to_prompt_with_semantic_parse(
                question, events, table_schema
            )

            prompt = self.process_question_to_prompt(question, table_schema)

            query = text(
                f"""
                              INSERT INTO {self.table_name}_text2sql (question, question_level, prompt, prompt_semantic)
                              VALUES (:question, :question_level, :prompt, :prompt_semantic)
                          """
            )

            with self.engine.connect() as connection:
                connection.execute(
                    query,
                    {
                        "question": question,
                        "question_level": row["question_level"],
                        "prompt": prompt,
                        "prompt_semantic": prompt_semantic,
                    },
                )

                connection.commit()

        prompt_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_text2sql",
            self.engine,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = []
            for index, row in prompt_df.iterrows():
                question = row["question"]

                progress_check_key = (
                    "sql_query" if not semantic_parse else "sql_query_semantic"
                )

                if (
                    row[progress_check_key] is not None
                    and row[progress_check_key] != ""
                ):
                    continue
                if semantic_parse:
                    prompt = row["prompt_semantic"]
                else:
                    prompt = row["prompt"]
                futures.append(
                    executor.submit(
                        self.text2sql_generation,
                        prompt=prompt,
                        question=question,
                        model_name="gpt-3.5-turbo",
                        semantic_parse=semantic_parse,
                    )
                )

        self.verify_results(semantic_parse=semantic_parse)

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
        Generate the sql query to retrieve the relevant information from the table to answer the question.
        Return all columns for the rows that satisfy the condition.
        Return the SQL query  in json format with the key "sql_query"
        """
        return prompt

    def process_question_to_prompt_with_semantic_parse(
        self, question: str, events: List, table_schema: str
    ):
        # TODO: and question mark, should we do this? and How?
        """
        Process the question to the prompt

        Args:
            question: The question
            events: The events
            table_schema: The table schema

        Returns:
            The prompt
        """

        related_entities = []

        for event in events:
            items = event.split("|")
            if len(items) != 5:
                continue
            subject, predicate, tail_object, start_time, end_time = event.split("|")
            if subject in question:
                related_entities.append(subject)
            if tail_object in question:
                related_entities.append(tail_object)
        related_entities = ",".join(related_entities)
        prompt = f"""question: {question}
        The related knowledge to answer this question is in table {table_schema},
        the table name is {self.table_name},        
        entities can be used as where clause: {related_entities}
        Generate the sql query to retrieve the relevant information from the table to answer the question.
        Return all columns for the rows that satisfy the condition.
        Return the SQL query in json format with the key "sql_query"
        """
        return prompt

    def text2sql_generation(
        self,
        prompt: str,
        question: str,
        model_name: str = "gpt-3.5-turbo",
        semantic_parse: bool = False,
    ):
        """
        This function will call text2sql_gpt and update the sql_query in the database
        Args:
            prompt: The prompt
            question: The question
            model_name: The model name
            semantic_parse: If True, use the semantic parse to generate the prompt
        """
        sql_query = self.text2sql_gpt(prompt=prompt, model_name=model_name)
        self.update_sql_query(
            question=question, sql_query=sql_query, semantic_parse=semantic_parse
        )

    @backoff.on_exception(
        backoff.constant, RateLimitError, raise_on_giveup=True, interval=20
    )
    def text2sql_gpt(self, prompt: str, model_name: str = "gpt-3.5-turbo"):
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
            raise e

    def update_sql_query(
        self, question: str, sql_query: str, semantic_parse: bool = False
    ):
        if semantic_parse:
            query = text(
                f"""
                      UPDATE {self.table_name}_text2sql
                      SET sql_query_semantic = :sql_query
                      WHERE question = :question
                  """
            )
        else:
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
                    "sql_query": sql_query,
                },
            )
            connection.commit()

    def verify_results(self, semantic_parse: bool = False):
        if semantic_parse:
            prompts_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_text2sql WHERE sql_query_semantic IS NOT NULL",
                self.engine,
            )
        else:
            prompts_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_text2sql WHERE sql_query IS NOT NULL",
                self.engine,
            )
        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions",
            self.engine,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = []
            for index, row in prompts_df.iterrows():
                progress_key = "correct" if not semantic_parse else "correct_semantic"
                if row[progress_key] is not None:
                    continue
                futures.append(
                    executor.submit(
                        self.verify_one_result,
                        row=row,
                        questions_df=questions_df,
                        semantic_parse=semantic_parse,
                    )
                )

    def verify_one_result(self, row, questions_df, semantic_parse: bool = False):
        connection = self.engine.connect()
        question = row["question"]
        if semantic_parse:
            sql_query = row["sql_query_semantic"]
        else:
            sql_query = row["sql_query"]
        progress_key = "correct" if not semantic_parse else "correct_semantic"
        if row[progress_key] is not None:
            return
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

        if semantic_parse:
            query = text(
                f"""
                      UPDATE {self.table_name}_text2sql
                      SET correct_semantic = :correct
                      WHERE question = :question
                  """
            )
        else:
            query = text(
                f"""
                              UPDATE {self.table_name}_text2sql
                              SET correct = :correct
                              WHERE question = :question
                          """
            )

        connection.execute(
            query,
            {
                "question": question,
                "correct": correct,
            },
        )
        connection.commit()
        connection.close()

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
        table_name="unified_kg_cron",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )

    with timer(logger, "Benchmarking"):
        rag.benchmark(semantic_parse=True)
