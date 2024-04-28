import psycopg2
import json
from tkgqa_generator.utils import get_logger
import numpy as np
import pandas as pd
from datetime import datetime

logger = get_logger(__name__)


class TKGQA_GENERATOR:
    """
    type of the statement we will have includes
    - Timestamp
        - RE: Retrieval
        - RA: Reasoning
            - 2RA
            - 3RA-R
            - 3RA-A

    - Duration
        - RE: Retrieval
        - RA: Reasoning
            - 2RA
            - 3RA-R
            - 3RA-A


    input will be a unified knowledge graph, it will be stored in a table
        subject
        subject_json
        predicate
        predicate_json
        object
        object_json
        start_time
        end_time
    """

    def __init__(self,
                 table_name: str,
                 host: str,
                 port: int,
                 user: str,
                 password: str,
                 db_name: str = "tkgqa"
                 ):
        # setup the db connection
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.db_name
        )
        self.unified_kg_table = table_name
        # we also need to create a new table, we can call it
        self.unified_kg_table_statement = f"{self.unified_kg_table}_statement"
        # within the table, you will have the field: measurement, type, statement(json)
        self.cursor = self.connection.cursor()
        # create the table if not exists
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.unified_kg_table_statement} (
                id SERIAL PRIMARY KEY,
                measurement VARCHAR(255),
                type VARCHAR(255),
                statement JSONB
            )
        """)

    @staticmethod
    def allen_temporal_relation(start_time1, end_time1, start_time2, end_time2):
        """
        This function will return the allen temporal relation between two time ranges

        We have the 26 possible relations
        - 13 for time range operation
        - 10 for time point and time range operation
        - 3 for time point operation

        We will need to extract them from the quantitatively time range

        We will have "beginning of time" or "end of time" to represent the infinite time range
        We will need to convert it to a numerical value in np.inf
        Ohters will be converted to a numerical value in the timestamp

        :param start_time1:
        :param end_time1:
        :param start_time2:
        :param end_time2:
        :return:
        """
        if start_time1 == "beginning of time":
            start_time1 = datetime.min.replace(year=1)
        if end_time1 == "end of time":
            end_time1 = datetime.max.replace(year=9999)
        if start_time2 == "beginning of time":
            start_time2 = datetime.min.replace(year=1)
        if end_time2 == "end of time":
            end_time2 = datetime.max.replace(year=9999)

        # convert the time to numerical value, format is like this: 1939-04-25
        start_time1 = np.datetime64(start_time1)
        end_time1 = np.datetime64(end_time1)
        start_time2 = np.datetime64(start_time2)
        end_time2 = np.datetime64(end_time2)

        logger.info(
            f"start_time1: {start_time1}, end_time1: {end_time1}, start_time2: {start_time2}, end_time2: {end_time2}")

        # 13 for time range operation

    def timestamp_retrieval(self):
        """
        This function will generate a timestamp retrieval statement
        the statement should also be annotated, and stored in the database

        Because this is one hop retrieval, so we basically can loop the whole table, and generate the statement
        Do not need to consider the json field now

        Output format will be:
        {
            "statement": "xxx",
            "subject": "xxx",
            "predicate": "xxx",
            "object": "xxx",
            "start_time": "xxx",
            "end_time": "xxx"
        }
        """
        self.cursor.execute(f"SELECT * FROM {self.unified_kg_table}")
        results = self.cursor.fetchall()
        for result in results:
            # get result to dict, and extract the subject, predicate, object
            result_dict = {
                "subject": result[0],
                "predicate": result[2],
                "object": result[4],
                "start_time": result[6],
                "end_time": result[7]
            }
            statement = f"{result_dict['subject']} {result_dict['predicate']} {result_dict['object']} from {result_dict['start_time']} to {result_dict['end_time']}"

            result_dict = {
                "statement": statement,
                "subject": result_dict['subject'],
                "predicate": result_dict['predicate'],
                "object": result_dict['object'],
                "start_time": result_dict['start_time'],
                "end_time": result_dict['end_time']
            }

            # Serialize the statement dictionary to a JSON string
            json_statement = json.dumps(result_dict)

            # Store the statement
            measurement = 'timestamp'
            type_value = 'RE'

            # Execute the SQL command with the serialized JSON string
            self.cursor.execute(
                f"INSERT INTO {self.unified_kg_table_statement} (measurement, type, statement) VALUES (%s, %s, %s)",
                (measurement, type_value, json_statement)
            )
        self.connection.commit()

        """
        then is to generate the question and answer, it will be stored in another table
        notes: tried public available ones, not work well.
        so we have two ways to do this:
        1. dump it to LLM, and generate the question and answer
        2. Created it based on the template
        
        We will try the second one first, and then mix the two methods, so we can have different opinions
        """

    def timestamp_2ra(self):
        """
        This function will generate a timestamp 2 reasoning statement
        For example, we have two statement, we can ask which one happenend first
        So the statement will be SPO1 happened Temporal Operation SPO2

        There are several things we need to decide before we go forward:

        Do we loop all possible SPO combinations? We can try this wya first
        - It can be same subject, different predicate, different object
        - It can be different subject, same object

        So first step we will query from the unifed graph to grab our first SPO item
        Then we query the second one (they should be different)
        Then we generate the statement based on the two items
        """
        first_spo_df = pd.read_sql_query(f"SELECT * FROM {self.unified_kg_table}", self.connection)
        second_spo_df = first_spo_df.copy(deep=True)
        for first_index, first_spo in first_spo_df.iterrows():
            for second_index, second_spo in second_spo_df.iterrows():
                if first_index == second_index:
                    continue
                logger.info(f"first_spo: {first_spo}, second_spo: {second_spo}")
                self.allen_temporal_relation(
                    first_spo['start_time'],
                    first_spo['end_time'],
                    second_spo['start_time'],
                    second_spo['end_time']
                )
                break
            break


if __name__ == "__main__":
    generator = TKGQA_GENERATOR(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa"
    )
    generator.timestamp_2ra()
