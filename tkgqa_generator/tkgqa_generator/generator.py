import json
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

from tkgqa_generator.utils import get_logger

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

    def __init__(
            self,
            table_name: str,
            host: str,
            port: int,
            user: str,
            password: str,
            db_name: str = "tkgqa",
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
            dbname=self.db_name,
        )
        self.unified_kg_table = table_name
        # we also need to create a new table, we can call it
        self.unified_kg_table_statement = f"{self.unified_kg_table}_statement"
        # within the table, you will have the field: measurement, type, statement(json)
        self.cursor = self.connection.cursor()
        # create the table if not exists
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.unified_kg_table_statement} (
                id SERIAL PRIMARY KEY,
                measurement VARCHAR(255),
                type VARCHAR(255),
                statement JSONB
            )
        """
        )

    @staticmethod
    def allen_temporal_relation(time_range_a, time_range_b):
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
        start_time1, end_time1 = time_range_a
        start_time2, end_time2 = time_range_b
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
            f"start_time1: {start_time1}, end_time1: {end_time1}, start_time2: {start_time2}, end_time2: {end_time2}"
        )

        # 13 for time range operation
        time_range_a_datetime = [start_time1, end_time1]
        time_range_b_datetime = [start_time2, end_time2]
        # then we will do the operation for the time range, get the allen temporal relation
        """
        x_start <= x_end
        y_start <= y_end
        allen_operator = [
            x_start - x_end,  # 0, or -1, which means a is a point or a range
            y_start - y_end, # 0, or -1
            x_start - y_start,
            x_start - y_end,
            x_end - y_start,
            x_end - y_end,
        ]
        
        After this do a operation for the allen_operator, if value = 0, keep it, < 0, set it to -1, > 0, set it to 1
        
        Then we will have:
        13 for time range operation, which means x_start < x_end, y_start < y_end
            - X <  Y => [-1, -1, -1, -1, -1, -1]
            - X m  Y => [-1, -1, -1, -1,  0, -1]
            - X o  Y => [-1, -1, -1, -1,  1, -1]
            - X fi Y => [-1, -1, -1, -1,  1,  0]
            - X di Y => [-1, -1, -1, -1,  1,  1]
            - X s  Y => [-1, -1,  0, -1,  1, -1]
            - X =  Y => [-1, -1,  0, -1,  1,  0]
            - X si Y => [-1, -1,  0, -1,  1,  1]
            - X d  Y => [-1, -1,  1, -1,  1, -1]
            - X f  Y => [-1, -1,  1, -1,  1,  0]
            - X oi Y => [-1, -1,  1, -1,  1,  1]
            - X mi Y => [-1, -1,  1,  0,  1,  1]
            - X >  Y => [-1, -1,  1,  1,  1,  1]
            
        10 for time point and time range operation
        Amony the 10, 5 for X is a point, 5 for Y is a point
        5 for X is a point, Y is a range, which means x_start = x_end, y_start < y_end
            - X <  Y => [0, -1, -1, -1, -1, -1]
            - X s  Y => [0, -1,  0, -1,  0, -1]
            - X d  Y => [0, -1,  1, -1,  1, -1]
            - X f  Y => [0, -1,  1,  0,  1,  0]
            - X >  Y => [0, -1,  1,  1,  1,  1]
        5 for X is a range, Y is a point, which means x_start < x_end, y_start = y_end
            - X <  Y => [-1, 0, -1, -1, -1, -1]
            - X fi Y => [-1, 0, -1，-1,  0,  0]
            - X di Y => [-1, 0, -1, -1,  1,  1]
            - X si Y => [-1, 0,  0,  0,  1,  1]
            - X >  Y => [-1, 0,  1,  1,  1,  1]
        
        3 for time point operation, which means x_start = x_end, y_start = y_end
            - X < Y => [0, 0, -1, -1, -1, -1]
            - X = Y => [0, 0,  0,  0,  0,  0]
            - X > Y => [0, 0,  1,  1,  1,  1]
        """

        ALLEN_OPERATOR_DICT = {
            (-1, -1, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tr",
                "code": "tr-1"
            },
            (-1, -1, -1, -1, 0, -1): {
                "relation": "X m Y",
                "description": "X meets Y",
                "category": "tr",
                "code": "tr-2"
            },
            (-1, -1, -1, -1, 1, -1): {
                "relation": "X o Y",
                "description": "X overlaps Y",
                "category": "tr",
                "code": "tr-3"
            },
            (-1, -1, -1, -1, 1, 0): {
                "relation": "X fi Y",
                "description": "X finishes Y",
                "category": "tr",
                "code": "tr-4"
            },
            (-1, -1, -1, -1, 1, 1): {
                "relation": "X di Y",
                "description": "X during Y",
                "category": "tr",
                "code": "tr-5"
            },
            (-1, -1, 0, -1, 1, -1): {
                "relation": "X s Y",
                "description": "X starts Y",
                "category": "tr",
                "code": "tr-6"
            },
            (-1, -1, 0, -1, 1, 0): {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "tr",
                "code": "tr-7"
            },
            (-1, -1, 0, -1, 1, 1): {
                "relation": "X si Y",
                "description": "X starts Y",
                "category": "tr",
                "code": "tr-8"
            },
            (-1, -1, 1, -1, 1, -1): {
                "relation": "X d Y",
                "description": "X during Y",
                "category": "tr",
                "code": "tr-9"
            },
            (-1, -1, 1, -1, 1, 0): {
                "relation": "X f Y",
                "description": "X finishes Y",
                "category": "tr",
                "code": "tr-10"
            },
            (-1, -1, 1, -1, 1, 1): {
                "relation": "X oi Y",
                "description": "X overlaps Y",
                "category": "tr",
                "code": "tr-11"
            },
            (-1, -1, 1, 0, 1, 1): {
                "relation": "X mi Y",
                "description": "X meets Y",
                "category": "tr",
                "code": "tr-12"
            },
            (-1, -1, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tr",
                "code": "tr-13"
            },
            (0, -1, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tp&tr",
                "code": "tptr-14"
            },
            (0, -1, 0, -1, 0, -1): {
                "relation": "X s Y",
                "description": "X starts Y",
                "category": "tp&tr",
                "code": "tptr-15"
            },
            (0, -1, 1, -1, 1, -1): {
                "relation": "X d Y",
                "description": "X during Y",
                "category": "tp&tr",
                "code": "tptr-16"
            },
            (0, -1, 1, 0, 1, 0): {
                "relation": "X f Y",
                "description": "X finishes Y",
                "category": "tp&tr",
                "code": "tptr-17"
            },
            (0, -1, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tp&tr",
                "code": "tptr-18"
            },
            (-1, 0, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tr&tp",
                "code": "trtp-19"
            },
            (-1, 0, -1, -1, 0, 0): {
                "relation": "X fi Y",
                "description": "X finishes Y",
                "category": "tr&tp",
                "code": "trtp-20"
            },
            (-1, 0, -1, -1, 1, 1): {
                "relation": "X di Y",
                "description": "X during Y",
                "category": "tr&tp",
                "code": "trtp-21"
            },
            (-1, 0, 0, 0, 1, 1): {
                "relation": "X si Y",
                "description": "X starts Y",
                "category": "tr&tp",
                "code": "trtp-22"
            },
            (-1, 0, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tr&tp",
                "code": "trtp-23"
            },
            (0, 0, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tp",
                "code": "tp-24"
            },
            (0, 0, 0, 0, 0, 0): {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "tp",
                "code": "tp-25"
            },
            (0, 0, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tp",
                "code": "tp-26"
            }
        }

        allen_operator = [
            time_range_a_datetime[0] - time_range_a_datetime[1],
            time_range_b_datetime[0] - time_range_b_datetime[1],
            time_range_a_datetime[0] - time_range_b_datetime[0],
            time_range_a_datetime[0] - time_range_b_datetime[1],
            time_range_a_datetime[1] - time_range_b_datetime[0],
            time_range_a_datetime[1] - time_range_b_datetime[1],
        ]

        # do the operation for the allen_operator
        for index, value in enumerate(allen_operator):
            if value == 0:
                allen_operator[index] = 0
            elif value < 0:
                allen_operator[index] = -1
            else:
                allen_operator[index] = 1

        # logger.critical(f"allen_operator: {allen_operator}")
        # get it to be a tuple
        allen_operator = tuple(allen_operator)
        logger.critical(f"allen_operator: {allen_operator}")
        logger.critical(f"ALLEN_OPERATOR_DICT: {ALLEN_OPERATOR_DICT[allen_operator]}")
        return ALLEN_OPERATOR_DICT[allen_operator]

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
                "end_time": result[7],
            }
            statement = f"{result_dict['subject']} {result_dict['predicate']} {result_dict['object']} from {result_dict['start_time']} to {result_dict['end_time']}"

            result_dict = {
                "statement": statement,
                "subject": result_dict["subject"],
                "predicate": result_dict["predicate"],
                "object": result_dict["object"],
                "start_time": result_dict["start_time"],
                "end_time": result_dict["end_time"],
            }

            # Serialize the statement dictionary to a JSON string
            json_statement = json.dumps(result_dict)

            # Store the statement
            measurement = "timestamp"
            type_value = "RE"

            # Execute the SQL command with the serialized JSON string
            self.cursor.execute(
                f"INSERT INTO {self.unified_kg_table_statement} (measurement, type, statement) VALUES (%s, %s, %s)",
                (measurement, type_value, json_statement),
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
        first_spo_df = pd.read_sql_query(
            f"SELECT * FROM {self.unified_kg_table}", self.connection
        )
        second_spo_df = first_spo_df.copy(deep=True)
        for first_index, first_spo in first_spo_df.iterrows():
            for second_index, second_spo in second_spo_df.iterrows():
                if first_index == second_index:
                    continue
                # logger.info(f"first_spo: {first_spo}, second_spo: {second_spo}")
                allen_temporal_rel = self.allen_temporal_relation(
                    [first_spo["start_time"], first_spo["end_time"]],
                    [second_spo["start_time"], second_spo["end_time"]],
                )
                # save the statement


if __name__ == "__main__":
    generator = TKGQA_GENERATOR(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    generator.timestamp_2ra()
