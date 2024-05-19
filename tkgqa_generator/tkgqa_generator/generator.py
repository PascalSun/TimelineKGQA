import json
from datetime import datetime

import numpy as np
import pandas as pd
from tkgqa_generator.openai_utils import paraphrase_question
import psycopg2

from tkgqa_generator.utils import get_logger

logger = get_logger(__name__)


class TKGQAGenerator:
    """
    type of the statement we will have includes
    - Timestamp
        - RE: Retrieval
        - RA: Reasoning
            - 2RA-Allen
            - 2RA-Set
            - 2RA-Aggregation

    - Duration
        - RE: Retrieval
        - RA: Reasoning
            - 2RA-Allen
            - 2RA-Aggeration


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
                statement JSONB,
                questions JSONB
            )
        """
        )

    @staticmethod
    def retrieve_tr(s, p, o, start_time, end_time) -> dict:
        """
        This will try to generate four questions belong to RE type

        The questions will be:
        - ? p o during the time range from start_time to end_time?
        - s p ? during the time range from start_time to end_time?
        - s p o from ? to end_time?
        - s p o from start_time to ?
        - s p o from ? to ?

        Args:
            s (str): The subject
            p (str): The predicate
            o (str): The object
            start_time (datetime): The start time
            end_time (datetime): The end time

        :return:
            dict: The generated questions
        """
        questions = {
            "?potstd": {
                "q": f"? {p} {o} during the time range from {start_time} to {end_time}?",
                "a": f"{s}",
            },
            "sp?tstd": {
                "q": f"{s} {p} ? during the time range from {start_time} to {end_time}?",
                "a": f"{o}",
            },
            "spo?td": {
                "q": f"{s} {p} {o} from ? to {end_time}?",
                "a": f"{start_time}",
            },
            "spots?": {
                "q": f"{s} {p} {o} from {start_time} to ?",
                "a": f"{end_time}",
            },
            "spo??": {
                "q": f"{s} {p} {o} from ? to ?",
                "a": f"{start_time} and {end_time}",
            },
        }
        logger.info(f"questions: {questions}")
        # we will need to feed the questions to LLM, generate proper question statement
        for question_type, question_dict in questions.items():
            question = question_dict["q"]
            answer = question_dict["a"]
            paraphrased_question = paraphrase_question(question)
            logger.info(f"paraphrased_question: {paraphrased_question}")
            question_dict["pq"] = paraphrased_question
        logger.info(f"questions: {questions}")
        return questions

    @staticmethod
    def allen_tr_relation(
            time_range_a: list[datetime, datetime], time_range_b: list[datetime, datetime]
    ) -> dict:
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

        Args:
            time_range_a (list[datetime, datetime]): The first time range
            time_range_b (list[datetime, datetime]): The second time range

        Returns:
            dict: The allen temporal relation between the two time ranges
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
                "code": "tr-1",
            },
            (-1, -1, -1, -1, 0, -1): {
                "relation": "X m Y",
                "description": "X meets Y",
                "category": "tr",
                "code": "tr-2",
            },
            (-1, -1, -1, -1, 1, -1): {
                "relation": "X o Y",
                "description": "X overlaps Y",
                "category": "tr",
                "code": "tr-3",
            },
            (-1, -1, -1, -1, 1, 0): {
                "relation": "X fi Y",
                "description": "X finishes Y",
                "category": "tr",
                "code": "tr-4",
            },
            (-1, -1, -1, -1, 1, 1): {
                "relation": "X di Y",
                "description": "X during Y",
                "category": "tr",
                "code": "tr-5",
            },
            (-1, -1, 0, -1, 1, -1): {
                "relation": "X s Y",
                "description": "X starts Y",
                "category": "tr",
                "code": "tr-6",
            },
            (-1, -1, 0, -1, 1, 0): {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "tr",
                "code": "tr-7",
            },
            (-1, -1, 0, -1, 1, 1): {
                "relation": "X si Y",
                "description": "X starts Y",
                "category": "tr",
                "code": "tr-8",
            },
            (-1, -1, 1, -1, 1, -1): {
                "relation": "X d Y",
                "description": "X during Y",
                "category": "tr",
                "code": "tr-9",
            },
            (-1, -1, 1, -1, 1, 0): {
                "relation": "X f Y",
                "description": "X finishes Y",
                "category": "tr",
                "code": "tr-10",
            },
            (-1, -1, 1, -1, 1, 1): {
                "relation": "X oi Y",
                "description": "X overlaps Y",
                "category": "tr",
                "code": "tr-11",
            },
            (-1, -1, 1, 0, 1, 1): {
                "relation": "X mi Y",
                "description": "X meets Y",
                "category": "tr",
                "code": "tr-12",
            },
            (-1, -1, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tr",
                "code": "tr-13",
            },
            (0, -1, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tp&tr",
                "code": "tptr-14",
            },
            (0, -1, 0, -1, 0, -1): {
                "relation": "X s Y",
                "description": "X starts Y",
                "category": "tp&tr",
                "code": "tptr-15",
            },
            (0, -1, 1, -1, 1, -1): {
                "relation": "X d Y",
                "description": "X during Y",
                "category": "tp&tr",
                "code": "tptr-16",
            },
            (0, -1, 1, 0, 1, 0): {
                "relation": "X f Y",
                "description": "X finishes Y",
                "category": "tp&tr",
                "code": "tptr-17",
            },
            (0, -1, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tp&tr",
                "code": "tptr-18",
            },
            (-1, 0, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tr&tp",
                "code": "trtp-19",
            },
            (-1, 0, -1, -1, 0, 0): {
                "relation": "X fi Y",
                "description": "X finishes Y",
                "category": "tr&tp",
                "code": "trtp-20",
            },
            (-1, 0, -1, -1, 1, 1): {
                "relation": "X di Y",
                "description": "X during Y",
                "category": "tr&tp",
                "code": "trtp-21",
            },
            (-1, 0, 0, 0, 1, 1): {
                "relation": "X si Y",
                "description": "X starts Y",
                "category": "tr&tp",
                "code": "trtp-22",
            },
            (-1, 0, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tr&tp",
                "code": "trtp-23",
            },
            (0, 0, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tp",
                "code": "tp-24",
            },
            (0, 0, 0, 0, 0, 0): {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "tp",
                "code": "tp-25",
            },
            (0, 0, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tp",
                "code": "tp-26",
            },
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

    @staticmethod
    def allen_td_relation(
            time_range_a: list[datetime, datetime], time_range_b: list[datetime, datetime]
    ) -> dict:
        """

        Args:
            time_range_a (list[datetime, datetime]): The first time range
            time_range_b (list[datetime, datetime]): The second time range

        Returns:
            dict: The allen temporal relation between the two time ranges
        """
        duration_a = abs(time_range_a[1] - time_range_a[0])
        duration_b = abs(time_range_b[1] - time_range_b[0])
        if duration_a < duration_b:
            return {
                "relation": "X < Y",
                "description": "X is shorter Y",
                "category": "td",
                "code": "td-1",
            }
        elif duration_a == duration_b:
            return {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "td",
                "code": "td-2",
            }
        else:
            return {
                "relation": "X > Y",
                "description": "X is longer Y",
                "category": "td",
                "code": "td-3",
            }

    @staticmethod
    def set_operator(
            time_range_a, time_range_b: list = None, temporal_operator: str = None
    ) -> set:
        """
        This function will return the temporal operator between two time ranges
        The temporal operator can be:
        - Intersection
        - Union
        - Complement

        Args:
            time_range_a (list): The first time range
            time_range_b (list): The second time range
            temporal_operator (str): The temporal operator

        It is basically try to output a set of new time range.

        Returns:
            set: The new time range

        """
        if temporal_operator is None or temporal_operator not in [
            "intersection",
            "union",
            "complement",
        ]:
            raise ValueError(
                "temporal_operator should be one of the following: intersection, union, complement"
            )
        if temporal_operator == "intersection":
            # use set do it directly
            intersection = set(time_range_a) & set(time_range_b)
            return intersection
        elif temporal_operator == "union":
            # use set do it directly
            union = set(time_range_a) | set(time_range_b)
            return union
        elif temporal_operator == "complement":
            """
            This should operate on one time range
            """
            if time_range_b is not None:
                raise ValueError(
                    "time_range_b should be None when temporal_operator is complement"
                )
            # use set do it directly
            universal_time_range = set(
                [datetime.min.replace(year=1), datetime.max.replace(year=9999)]
            )
            complement = universal_time_range - set(time_range_a)
            return complement
        else:
            raise ValueError(
                "temporal_operator should be one of the following: intersection, union, complement"
            )

    @staticmethod
    def aggregate_tr_operator(
            time_ranges: list[[datetime, datetime]], agg_temporal_operator: str = None
    ) -> list:
        """
        For the time range, it will do the rank operation, sort it

        Aggregation operator can be:
        - ranking(min, max)
            - ranking_start
            - ranking_end

        Args:
            time_ranges (list): The list of time ranges
            agg_temporal_operator (str): The aggregation temporal operator

        Returns:
            list: the list of sorted index for the time range

        For example we have the time range:

        ```
        time_ranges = [
            (datetime(2023, 5, 1, 12, 0), datetime(2023, 5, 1, 15, 0)),  # 3 hours
            (datetime(2023, 5, 1, 9, 30), datetime(2023, 5, 1, 14, 0)),  # 4.5 hours
            (datetime(2023, 5, 1, 8, 0), datetime(2023, 5, 1, 11, 30)),  # 3.5 hours
            (datetime(2023, 5, 2, 9, 30), datetime(2023, 5, 2, 12, 0)),  # 2.5 hours
            (datetime(2023, 5, 1, 10, 30), datetime(2023, 5, 1, 13, 0))  # 2.5 hours
        ]

        result_start = TKGQAGenerator.aggregate_tr_temporal_operator(time_ranges, "ranking_start")
        [3,1,0,4,2]

        result_end = TKGQAGenerator.aggregate_tr_temporal_operator(time_ranges, "ranking_end")
        [2,4,3,0,1]
        ```
        """

        # Create a list of indices paired with time ranges
        indexed_time_ranges = list(enumerate(time_ranges))

        if agg_temporal_operator == "ranking_start":
            # Sort by start time, but maintain original indices
            indexed_time_ranges.sort(key=lambda x: x[1][0])
        elif agg_temporal_operator == "ranking_end":
            # Sort by end time, but maintain original indices
            indexed_time_ranges.sort(key=lambda x: x[1][1])
        else:
            raise ValueError(
                "Unsupported aggregation temporal operator. Please use 'ranking_start' or 'ranking_end'."
            )

        # After sorting, create a new list that maps the original index to its new rank
        rank_by_index = [0] * len(time_ranges)  # Pre-initialize a list of zeros
        for rank, (original_index, _) in enumerate(indexed_time_ranges):
            rank_by_index[original_index] = rank

        return rank_by_index

    @staticmethod
    def aggregate_td_operator(
            time_ranges: list[[datetime, datetime]], agg_temporal_operator: str = None
    ) -> list:
        """
        For the time range, it will do the rank operation, sort it

        First calculate the duration of the time range, then do the rank operation based on the duration

        Args:
            time_ranges (list): The list of time ranges
            agg_temporal_operator (str): The aggregation temporal operator

        Returns:
            list: the list of sorted index for the time range


        Example:
        ```
        time_ranges = [
            (datetime(2023, 5, 1, 12, 0), datetime(2023, 5, 1, 15, 0)),  # 3 hours
            (datetime(2023, 5, 1, 9, 30), datetime(2023, 5, 1, 14, 0)),  # 4.5 hours
            (datetime(2023, 5, 1, 8, 0), datetime(2023, 5, 1, 11, 30)),  # 3.5 hours
            (datetime(2023, 5, 2, 9, 30), datetime(2023, 5, 2, 12, 0)),  # 2.5 hours
            (datetime(2023, 5, 1, 10, 30), datetime(2023, 5, 1, 13, 0))  # 2.5 hours
        ]
        ```

        The output will be:
        ```
        [2, 4, 3, 0, 1]
        ```
        """
        # Create a list of indices paired with time ranges
        if agg_temporal_operator == "ranking":
            indexed_time_ranges = list(enumerate(time_ranges))

            indexed_time_ranges.sort(key=lambda x: abs(x[1][1] - x[1][0]))
            rank_by_index = [0] * len(time_ranges)  # Pre-initialize a list of zeros
            for index, (original_index, _) in enumerate(indexed_time_ranges):
                rank_by_index[original_index] = index
            return rank_by_index
        if agg_temporal_operator == "sum":
            # total value of the time range
            durations = [
                abs(time_range[1] - time_range[0]) for time_range in time_ranges
            ]
            return sum(durations)
        if agg_temporal_operator == "average":
            # average value of the time range
            durations = [
                abs(time_range[1] - time_range[0]) for time_range in time_ranges
            ]
            return sum(durations) / len(durations)
        raise ValueError(
            "Unsupported aggregation temporal operator. Please use 'ranking', 'sum' or 'average'."
        )

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
            questions = self.retrieve_tr(
                s=result_dict["subject"],
                p=result_dict["predicate"],
                o=result_dict["object"],
                start_time=result_dict["start_time"],
                end_time=result_dict["end_time"],
            )

            # Serialize the statement dictionary to a JSON string
            json_statement = json.dumps(result_dict)

            # Store the statement
            measurement = "timestamp"
            type_value = "RE"

            # Execute the SQL command with the serialized JSON string
            self.cursor.execute(
                f"INSERT INTO {self.unified_kg_table_statement} (measurement, type, statement, questions) VALUES (%s, %s, %s, %s)",
                (measurement, type_value, json_statement, json.dumps(questions)),
            )
            break
        self.connection.commit()

        """
        then is to generate the question and answer, it will be stored in another table
        notes: tried public available ones, not work well.
        so we have two ways to do this:
        1. dump it to LLM, and generate the question and answer
        2. Created it based on the template
        
        We will try the second one first, and then mix the two methods, so we can have different opinions
        """

    def timestamp_2ra_allen(self):
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
                logger.info(f"allen_temporal_rel: {allen_temporal_rel}")
                return


if __name__ == "__main__":
    generator = TKGQAGenerator(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    generator.timestamp_retrieval()
    # generator.timestamp_2ra_allen()
