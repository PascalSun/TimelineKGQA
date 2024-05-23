import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2

from tkgqa_generator.openai_utils import (
    paraphrase_medium_question,
    paraphrase_simple_question,
)
from tkgqa_generator.templates import QUESTION_TEMPLATES
from tkgqa_generator.utils import get_logger

logger = get_logger(__name__)


class TKGQAGenerator:
    """
    # How human handle the temporal information and answer the temporal questions?

    ## Information Indexing
    When we see something, for example, an accident happen near our home in today morning.
    We need to first index this event into our brain.
    As we live in a three dimension space together with a time dimension,
    when we want to store this in our memory, (we will treat our memory as a N dimension space)
    - Index the spatial dimensions: is this close to my home or close to one of the point of interest in my mind
    - Index the temporal dimension: Temporal have several aspects
        - Treat temporal as Straight Homogenous(Objective) Timeline: Exact date when it happen, for example, [2023-05-01 10:00:00, 2023-05-01 10:30:00]
        - Treat temporal as Cycle Homogenous(Objective) sTimeline: Monday, First day of Month, Spring, 21st Century, etc. (You can aslo cycle the timeline based on your own requirement)
        - Treat temporal as Straight Hoterogenous(Subjective) Timeline: If you sleep during night, it will be fast for you in the 8 hours, however, if someone is working overnight, time will be slow for him.
        - Treat temporal as Cycle Hoterogenous(Subjective) Timeline: Life has different turning points for everyone, until they reach the end of their life.
    - Then index the information part: What happen, who is involved, what is the impact, etc.

    So in summary, we can say that in our mind, if we treat the event as embedding,
    part of the embedding will represent the temporal dimension information,
    part of the embedding will represent the spatial dimension information,
    the rest of the embedding will represent the general information part.
    This will help us to retrieve the information when we need it.

    ## Information Retrieval
    So when we try to retrieval the information, espeically the temporal part of the information.
    Normally we have several types:

    - Timeline Recovery: When Bush starts his term as president of US?
        - First: **General Information Retrieval** [(Bush, start, president of US), (Bush, term, president of US)]
        - Second: **Timeline Recovery Retrieval** [(Bush, start, president of US, 2000, 2000), (Bush, term, president of US, 2000, 2008)]
        - Third: Answer the question based on the timeline information
    - Temporal Constrainted Retrieval: In 2009, who is the president of US?
        - First: **General Information Retrieval** [(Bush, president of US), (Obama, president of US), (Trump, president of US)]
        - Second: **Temporal Constraint Retrieval** [(Obama, president of US, 2009, 2016)]
        - Third: Answer the question based on the temporal constraint information

    Three key things here:
    - **General Information Retrieval**: Retrieve the general information from the knowledge graph based on the question
    - **Temporal Constrainted Retrieval**: Filter on general information retrieval, apply the temporal constraint
    - **Timeline Recovery Retrieval**: Based on general information retrieval, recover the timeline information

    ## Temporal Questions
    We can try to classify the temporal questions from quite a few perspectives:
    - Based on Answer: Entity, Temporal
    - Based on Temporal Relations in Question: Before, After, During , etc or First, Last, etc.
    - Based on Temporal Representation Type: Point, Range, Duration, etc.
    - Based on Complexity of Question: Simple (direct retrieval), Complex (Multiple hops with the three key things we mention above)

    There is still no agreement or clear classification here, most of them stays in the first two.
    However, it is obvious that they have overlaps, so will not be the best way to advance the temporal embedding algorithms development.

    We are trying to decompose the question into the three key parts we mentioned above, so we can evaluate the ability of the models for this three key capabilities.

    ### Simple: Timeline and One Event Involved
    - Timeline Recovery: When Bush starts his term as president of US?
        - General Information Retrieval => Timeline Recovery => Answer the question
        - Question Focus can be: Timestamp Start, Timestamp End, Duration, Timestamp Start and End
    - Temporal Constrainted Retrieval: In 2009, who is the president of US?
        - General Information Retrieval => Temporal Constraint Retrieval => Answer the question
        - Question Focus can be: Subject, Object, Predicate. Can be more complex if we want mask out more elements

    ### Medium: Timeline and Two Events Involved
    - Timeline Recovery + Timeline Recovery: Is Bush president of US when 911 happen?
        - (General Information Retrieval => Timeline Recovery) And (General Information Retrieval => Timeline Recovery) => Timeline Operation => Answer the question
        - Question Focus can be: A new Time Range, A temporal relation (Before, After, During, etc.), A list of Time Range (Ranking), or Comparison of Duration
    - Timeline Recovery + Temporal Constrainted Retrieval: When Bush is president of US, who is the president of China?
        - (General Information Retrieval => Timeline Recovery) => Temporal Constraint Retrieval => Answer the question
        - This is same as above, Question Focus can be: Subject, Object

    ### Complex: Timeline and Multiple Events Involved
    In general, question focus (answer type) will only be two types when we extend from Medium Level
    - Timeline Operation
    - (Subject, Predicate, Object)

    So if we say Complex is 3 events and Timeline.

    - Timeline Recovery + Timeline Recovery + Timeline Recovery: When Bush is president of US and Putin is President of Russion, is Hu the president of China?
        - (General Information Retrieval => Timeline Recovery) And (General Information Retrieval => Timeline Recovery) And (General Information Retrieval => Timeline Recovery) => Timeline Operation => Answer the question
    - Timeline Recovery + Timeline Recovery + Temporal Constrainted Retrieval: When Bush is president of US and Putin is President of Russion, who is the president of China?

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
        self.unified_kg_table_questions = f"{self.unified_kg_table}_questions"
        self.cursor = self.connection.cursor()
        # create a table to store retrieval questions
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.unified_kg_table_questions} (
                id SERIAL PRIMARY KEY,
                source_kg_id INTEGER,
                question VARCHAR(1024),
                answer VARCHAR(1024),
                paraphrased_question VARCHAR(1024),
                events VARCHAR(1024)[],
                question_level VARCHAR(1024),
                question_type VARCHAR(1024),
                answer_type VARCHAR(1024),
                temporal_relation VARCHAR(1024) DEFAULT NULL
            );
        """
        )

    def simple_question_generation(self):
        """
        ## Types of Questions
        This is used to generate the simple question, we will have two types of questions.

        For each type of questions, based on the answer or the question focus, we can further divide them into
        - Timeline Recovery
            - Start TimePoint
            - End TimePoint
            - Time Range
            - Duration
        - Temporal Constrainted Retrieval (Ignore predicate for now)
            - Subject
            - Object

        Simple: Timeline and One Event Involved
        - Timeline Recovery: When Bush starts his term as president of US?
            - General Information Retrieval => Timeline Recovery => Answer the question
            - Question Focus can be: Timestamp Start, Timestamp End, Duration, Timestamp Start and End
        - Temporal Constrainted Retrieval: In 2009, who is the president of US?
            - General Information Retrieval => Temporal Constraint Retrieval => Answer the question
            - Question Focus can be: Subject, Object, Predicate. Can be more complex if we want mask out more elements


        ## Templates
        To generate the questions, We can try to feed into the LLM, and generate the questions.
        However, the diversity of the questions is not guaranteed, so we can use the template to generate the questions.
        Then use LLM to pharaphrase the questions.

        Template examples:
        - Timeline Recovery
            - Start TimePoint: When did {subject} start the term as {object}?
            - End TimePoint: When did {subject} end the term as {object}?
            - Time Range: When did {subject} serve as {object}?
            - Duration: How long did {subject} serve as {object}?
        - Temporal Constrainted Retrieval
            - Subject:
                - Who is afficiated to {subject} from {timestamp start} to {timestamp end}?
                - Who is afficiated to {subject} in {timestamp}?
            - Object:
                - {subject} is afficiated to which organisation from {timestamp start} to {timestamp end}?
                - {subject} is afficiated to which organisation during {temporal representation}?

        ## Process
        - Extract {subject}, {predicate}, {object}, {start_time}, {end_time} from the unified graph
        - Generate the questions based on the template for each type
        - Use LLM to paraphrase the questions

        Output format will be:
        - {question}
        - {answer}
        - {paraphrased_question}
        - subject, predicate, object, start_time, end_time
        - {question_level} => Simple
        - {question_type} => Timeline Recovery, Temporal Constrainted Retrieval
        - {answer_type} => Subject, Object | Timestamp Start, Timestamp End, Duration, Timestamp Start and End
        """
        # get records not yet generated questions
        self.cursor.execute(
            f"SELECT * FROM {self.unified_kg_table} WHERE id not in (SELECT source_kg_id FROM {self.unified_kg_table_questions})"
        )
        events_df = pd.DataFrame(self.cursor.fetchall())
        # set the column names
        events_df.columns = [desc[0] for desc in self.cursor.description]

        for index, event in events_df.iterrows():
            logger.info(f"event: {event}")
            questions = self.simple_question_generation_individual(
                subject=event["subject"],
                predicate=event["predicate"],
                object=event["object"],
                start_time=event["start_time"],
                end_time=event["end_time"],
                template_based=True,
            )

            # insert each qa into the table, have a flat table
            for question_obj in questions:
                indiv_sql_command = """
                    INSERT INTO {} (source_kg_id, 
                                    question, 
                                    answer, 
                                    paraphrased_question,
                                    events,
                                    question_level,
                                    question_type,
                                    answer_type
                                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """.format(
                    self.unified_kg_table_questions
                )

                question_obj["source_kg_id"] = event["id"]
                # get dict to tuple, sequence should be the same as the sql command
                data = (
                    question_obj["source_kg_id"],
                    question_obj["question"],
                    question_obj["answer"],
                    question_obj["pharaphrased_question"],
                    question_obj["events"],
                    question_obj["question_level"],
                    question_obj["question_type"],
                    question_obj["answer_type"],
                )
                logger.info(f"question: {question_obj['pharaphrased_question']}")
                self.cursor.execute(indiv_sql_command, data)
                self.connection.commit()
            break
        self.connection.commit()

    @staticmethod
    def simple_question_generation_individual(
        subject: str,
        predicate: str,
        object: str,
        start_time: str,
        end_time: str,
        template_based: bool = False,
        pharaphrased: bool = False,
    ) -> dict:
        """
        This will try to generate four questions belong to RE type

        The questions will be:
        - ? p o during the time range from start_time to end_time?
        - s p ? during the time range from start_time to end_time?
        - s p o from ? to end_time?
        - s p o from start_time to ?
        - s p o from ? to ?
        - [How long/What's the duration/etc] ? for the statement s p o

        Args:
            subject (str): The subject
            predicate (str): The predicate
            object (str): The object
            start_time (str): The start time
            end_time (str): The end time
            template_based (bool): Whether use the template based question generation
            pharaphrased (bool): Whether do the paraphrase for the question, if set to False,
                    then the paraphrased_question will be the same as the question

        Returns:
            dict: The generated questions
                - question
                - answer
                - paraphrased_question
                - events
                - question_level: Simple
                - question_type: The type of the question
                - answer_type: The type of the answer
        """

        questions = [
            {
                "question": f"??? {predicate} {object} during the time range from {start_time} to {end_time}?",
                "answer": f"{subject}",
                "pharaphrased_question": None,
                "events": [f"{subject}|{predicate}|{object}|{start_time}|{end_time}"],
                "question_level": "simple",
                "question_type": "temporal_constrainted_retrieval",
                "answer_type": "subject",
            },
            {
                "question": f"{subject} {predicate} ??? during the time range from {start_time} to {end_time}?",
                "answer": f"{object}",
                "pharaphrased_question": None,
                "events": [f"{subject}|{predicate}|{object}|{start_time}|{end_time}"],
                "question_level": "simple",
                "question_type": "temporal_constrainted_retrieval",
                "answer_type": "object",
            },
            {
                "question": f"{subject} {predicate} {object} from ??? to {end_time}?",
                "answer": f"{start_time}",
                "pharaphrased_question": None,
                "events": [f"{subject}|{predicate}|{object}|{start_time}|{end_time}"],
                "question_level": "simple",
                "question_type": "timeline_recovery",
                "answer_type": "timestamp_start",
            },
            {
                "question": f"{subject} {predicate} {object} from {start_time} to ???",
                "answer": f"{end_time}",
                "pharaphrased_question": None,
                "events": [f"{subject}|{predicate}|{object}|{start_time}|{end_time}"],
                "question_level": "simple",
                "question_type": "timeline_recovery",
                "answer_type": "timestamp_end",
            },
            {
                "question": f"{subject} {predicate} {object} from ??? to ???",
                "answer": f"{start_time} and {end_time}",
                "pharaphrased_question": None,
                "events": [f"{subject}|{predicate}|{object}|{start_time}|{end_time}"],
                "question_level": "simple",
                "question_type": "timeline_recovery",
                "answer_type": "timestamp_range",
            },
            {
                "question": f"[How long/What's the duration/etc] ??? for the statement {subject} {predicate} {object}",
                "answer": f"{end_time} - {start_time}",
                "pharaphrased_question": None,
                "events": [f"{subject}|{predicate}|{object}|{start_time}|{end_time}"],
                "question_level": "simple",
                "question_type": "timeline_recovery",
                "answer_type": "duration",
            },
        ]
        if template_based:
            # we will random pick one from the template
            for question_draft in questions:
                this_type_templates = QUESTION_TEMPLATES[
                    question_draft["question_level"]
                ][question_draft["question_type"]][question_draft["answer_type"]]
                logger.info(f"this_type_templates: {this_type_templates}")
                random_pick_template = random.choice(this_type_templates)
                # replace {subject}, {predicate}, {object}, {start_time}, {end_time} with the real value
                question_draft["question"] = random_pick_template.format(
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    start_time=start_time,
                    end_time=end_time,
                )

        if pharaphrased:
            for question_obj in questions:
                paraphrased_question = paraphrase_simple_question(
                    question=question_obj["question"]
                )
                logger.info(f"paraphrased_question: {paraphrased_question}")
                question_obj["pharaphrased_question"] = paraphrased_question
        else:
            for question_obj in questions:
                question_obj["pharaphrased_question"] = question_obj["question"]
        return questions

    def medium_question_generation(self):
        """
        This will involve mainly two types of questions

        **Timeline Recovery => Temporal Constrainted Retrieval**
        **Timeline Recovery + Timeline Recovery**

        - question_level: medium
        - question_type:
            - timeline_recovery_temporal_constrainted_retrieval
            - timeline_recovery_timeline_recovery
        - answer_type:
            - entity:
                - subject:
                - object
            - temporal related
                - Infer a new time range: Union/Intersection
                - Infer a temporal relation: Allen
                - Infer a list of time ranges: Ranking

        """
        self.cursor.execute(f"SELECT * FROM {self.unified_kg_table}")
        first_event_df = pd.DataFrame(self.cursor.fetchall())
        first_event_df.columns = [desc[0] for desc in self.cursor.description]
        second_event_df = first_event_df.copy(deep=True)
        for first_index, first_event in first_event_df.iterrows():
            for second_index, second_event in second_event_df.iterrows():
                if first_index == second_index:
                    continue

                source_kg_id = first_event["id"] * 1000000 + second_event["id"]
                # check is this inthe datebase, if yes, then contine
                self.cursor.execute(
                    f"SELECT * FROM {self.unified_kg_table_questions} WHERE source_kg_id = {source_kg_id}"
                )
                if self.cursor.fetchone():
                    continue
                # logger.info(f"first_spo: {first_spo}, second_spo: {second_spo}")
                # TODO: can try to filter the events to make sure it make more sense in the final question
                questions = self.medium_question_generation_individual(
                    first_event=first_event.to_dict(),
                    second_event=second_event.to_dict(),
                    template_based=True,
                    pharaphrased=True,
                )
                for question_obj in questions:
                    indiv_sql_command = """
                    INSERT INTO {} (source_kg_id,
                                    question,
                                    answer,
                                    paraphrased_question,
                                    events,
                                    question_level,
                                    question_type,
                                    answer_type,
                                    temporal_relation
                                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """.format(
                        self.unified_kg_table_questions,
                    )

                    question_obj["source_kg_id"] = source_kg_id
                    # get dict to tuple, sequence should be the same as the sql command
                    data = (
                        question_obj["source_kg_id"],
                        question_obj["question"],
                        question_obj["answer"],
                        question_obj["pharaphrased_question"],
                        question_obj["events"],
                        question_obj["question_level"],
                        question_obj["question_type"],
                        question_obj["answer_type"],
                        question_obj["temporal_relation"],
                    )
                    try:
                        self.cursor.execute(indiv_sql_command, data)
                        self.connection.commit()
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        logger.info(data)

                # return

    def medium_question_generation_individual(
        self,
        first_event: dict,
        second_event: dict,
        template_based: bool = True,
        pharaphrased: bool = True,
    ) -> dict:
        """
        - question_type:
            - timeline_recovery_temporal_constrainted_retrieval
                - For this one, the logic/reasoning/math part will be like: **TimeRange** + Temporal Semantic Operation => "TimeRange**
                - Then the interesting part will be the Timeline Operation, we have mentioned serveral types of operations below.
                    - There are mostly from numeric to semantic perspective
                    - Here is the reverse process: name it Temporal Semantic Operation
                    - So this is trying to convert the temporal semantic representation to a numeric operation and then get a new operation.
            - timeline_recovery_timeline_recovery
                - For the logic/reasoning/math side, it actually is **TimeRange** vs **TimeRange** => Timeline Operation
                    - Get an way to ask about this comparision relations.
                    - So the question will mainly be about whether this relation is True, or which relation it is.
                    - For duration, we can ask about the duration of the two events, and then compare
                    - Or we can compare the event ranking based on the time range
            - there is another types: Three years before 2019,  who is the president of China? => It is a valid question, but nobody will in this way.
                - It will be normally classied into **simple**: in 2016, who is the president of China?
                - Or it will be something like: Three years before bush end the term, who is the president of China? => This will be classifed into **Medium**, and belong to the timeline_recovery_temporal_constrainted_retrieval
        - answer_type:
            - subject, object for timeline_recovery_temporal_constrainted_retrieval
                - subject
                - object
                - only focus on the first one, as the second will always become the first later
            - temporal related for timeline_recovery_timeline_recovery
                - Infer a new time range: Union/Intersection
                - Infer a temporal relation: Allen
                - Infer a list of time ranges: Ranking
                - Infer duration, and then compare

        Process:

        The quality of the question is not guaranteed by LLM directly if we just mask out the answer.
        So we will use the template to generate the questions, then use LLM to paraphrase the questions.

        Args:
            first_event (dict): The first event
            second_event (dict): The second event
            template_based (bool): Whether use the template based question generation
            pharaphrased (bool): Whether do the paraphrase for the question, if set to False,
                    then the paraphrased_question will be the same as the question

        Returns:
            dict: The generated questions
                - question
                - answer
                - paraphrased_question
                - events
                - question_level: Medium
                - question_type: The type of the question
                - answer_type: The type of the answer
        """
        first_event_subject = first_event["subject"]
        first_event_predicate = first_event["predicate"]
        first_event_object = first_event["object"]
        first_event_start_time = first_event["start_time"]
        first_event_end_time = first_event["end_time"]

        second_event_subject = second_event["subject"]
        second_event_predicate = second_event["predicate"]
        second_event_object = second_event["object"]
        second_event_start_time = second_event["start_time"]
        second_event_end_time = second_event["end_time"]

        (
            first_event_start_time_dt,
            first_event_end_time_dt,
            second_event_start_time_dt,
            second_event_end_time_dt,
        ) = self.util_str_to_datetime(
            [first_event_start_time, first_event_end_time],
            [second_event_start_time, second_event_end_time],
        )

        # first generate
        # timeline_recovery => temporal_constrainted_retrieval
        # this will ask for the subject or object in one of the event

        medium_type_1_a_questions = []
        questions = []
        """
        Timeline Recovery => Temporal Constrainted Retrieval Questions 
        """
        # NOTES: question here actually is not used, because we will replace it with the template.
        # It is putting there to get the idea about the types of questions we are generating
        """
        The key part of this type is:
        We need to cover as many temporal semantic operations as possible
        - Before, After, During, this is the most common one and shown in the literature
        - Starts from the same time, Ends at the same time, Meets, Overlap, this is another way to add the temporal condition (inspired by the allen logic)
        - Above are from allen temporal logic and intersection/union
        - We can also add the ranking ones, however, before/after is the same as first/last, under this category
        - Then the rest is the one for duration, question like 3 years before, 3 years after, etc.
        
        So we have main two types of questions here:
        - Relation: Before, After, During ｜ Starts from the same time, Ends at the same time, Meets, Overlap
            - calculate the relation first, then generated based on template
            - Before: Who is the president of US before the end of Bush's term?
            - After: Who is the president of US after the start of Bush's term?
            - Starts from the same time: Who and Bush start their term as father and President of US respectively at the same time?
            - Ends at the same time: Who and Bush end their term as father and President of US respectively at the same time?
            - Meets: ?
            - During: Who is the president of US during Bush's term?
            - Overlap: Bush as the president of US meets who when the guy become the father?
        - Duration: 3 years before, 3 years after, 3 years after the end, etc.
            - calculate the duration first, then generated based on template
            - 3 years before: Who is the president of US 3 years before the end of Bush's term?
            - 3 years after: Who is the president of US 3 years after the start of Bush's term?
            - 3 years after the end: Who is the president of US 3 years after the end of Bush's term?
            - 3 years after the start: Who is the president of US 3 years after the start of Bush's term?
            - meets/during/overlap hard to get a time point, so not considered here.
        """
        # ask for first subject
        medium_type_1_a_questions.append(
            {
                "question": f"??? {first_event_predicate} {first_event_object} [Timeline Operation on ({first_event_start_time}, {first_event_end_time}) vs ({second_event_start_time}, {second_event_end_time})] {second_event_subject} {second_event_predicate} {second_event_object}?",
                "answer": f"{first_event_subject}",
                "pharaphrased_question": None,
                "events": [
                    f"{first_event_subject}|{first_event_predicate}|{first_event_object}|{first_event_start_time}|{first_event_end_time}",
                    f"{second_event_subject}|{second_event_predicate}|{second_event_object}|{second_event_start_time}|{second_event_end_time}",
                ],
                "question_level": "medium",
                "question_type": "timeline_recovery_temporal_constrainted_retrieval",
                "answer_type": "subject",
                "temporal_relation": None,
            }
        )

        # ask for first object
        medium_type_1_a_questions.append(
            {
                "question": f"{first_event_subject} {first_event_predicate} ??? [Timeline Operation on ({first_event_start_time}, {first_event_end_time}) vs ({second_event_start_time}, {second_event_end_time})] {second_event_subject} {second_event_predicate} {second_event_object}?",
                "answer": f"{first_event_object}",
                "pharaphrased_question": None,
                "events": [
                    f"{first_event_subject}|{first_event_predicate}|{first_event_object}|{first_event_start_time}|{first_event_end_time}",
                    f"{second_event_subject}|{second_event_predicate}|{second_event_object}|{second_event_start_time}|{second_event_end_time}",
                ],
                "question_level": "medium",
                "question_type": "timeline_recovery_temporal_constrainted_retrieval",
                "answer_type": "object",
                "temporal_relation": None,
            }
        )
        questions += medium_type_1_a_questions

        """
        For duration before, duration after type  question
        """
        medium_type_1_b_questions = []
        # this will be added later when we process the questions with template

        """
        Timeline Recovery + Timeline Recovery Questions
        
        This one is mainly from numeric to temporal semantic
        
        - Infer a new time range: Union/Intersection
        - Infer a temporal relation: Allen
        - Infer a list of time ranges: Ranking (not considered here)
        - Infer duration, and then compare
        """
        # timeline_recovery + timeline_recovery
        medium_type_2_questions = []
        # ask for union/intersection of the time range

        medium_type_2_questions = [
            {
                "question": f"{first_event_subject} {first_event_predicate} {first_event_object} ???[Timeline Operation on ({first_event_start_time}, {first_event_end_time}) vs ({second_event_start_time}, {second_event_end_time})]??? {second_event_subject} {second_event_predicate} {second_event_object}?",
                "answer": f"Union/Intersection of the time range",
                "pharaphrased_question": None,
                "events": [
                    f"{first_event_subject}|{first_event_predicate}|{first_event_object}|{first_event_start_time}|{first_event_end_time}",
                    f"{second_event_subject}|{second_event_predicate}|{second_event_object}|{second_event_start_time}|{second_event_end_time}",
                ],
                "question_level": "medium",
                "question_type": "timeline_recovery_timeline_recovery",
                "answer_type": "relation_union_or_intersection",
                "temporal_relation": "intersection",
            },
            {
                "question": f"{first_event_subject} {first_event_predicate} {first_event_object} ???[Timeline Operation on ({first_event_start_time}, {first_event_end_time}) vs ({second_event_start_time}, {second_event_end_time})]??? {second_event_subject} {second_event_predicate} {second_event_object}?",
                "answer": f"Union/Intersection of the time range",
                "pharaphrased_question": None,
                "events": [
                    f"{first_event_subject}|{first_event_predicate}|{first_event_object}|{first_event_start_time}|{first_event_end_time}",
                    f"{second_event_subject}|{second_event_predicate}|{second_event_object}|{second_event_start_time}|{second_event_end_time}",
                ],
                "question_level": "medium",
                "question_type": "timeline_recovery_timeline_recovery",
                "answer_type": "relation_union_or_intersection",
                "temporal_relation": "union",
            },
            {
                "question": f"{first_event_subject} {first_event_predicate} {first_event_object} ???[Timeline Operation on ({first_event_start_time}, {first_event_end_time}) vs ({second_event_start_time}, {second_event_end_time})]??? {second_event_subject} {second_event_predicate} {second_event_object}?",
                "answer": f"Temporal Relation",
                "pharaphrased_question": None,
                "events": [
                    f"{first_event_subject}|{first_event_predicate}|{first_event_object}|{first_event_start_time}|{first_event_end_time}",
                    f"{second_event_subject}|{second_event_predicate}|{second_event_object}|{second_event_start_time}|{second_event_end_time}",
                ],
                "question_level": "medium",
                "question_type": "timeline_recovery_timeline_recovery",
                "answer_type": "relation_allen",
                "temporal_relation": None,
                # under this, random choose choices question or yes/no question
                # for yes/no question, we randomly generate yes or no answer questions.
            },
            {
                "question": f"{first_event_subject} {first_event_predicate} {first_event_object} ???[Timeline Operation on ({first_event_start_time}, {first_event_end_time}) vs ({second_event_start_time}, {second_event_end_time})]??? {second_event_subject} {second_event_predicate} {second_event_object}?",
                "answer": f"Duration",
                "pharaphrased_question": None,
                "events": [
                    f"{first_event_subject}|{first_event_predicate}|{first_event_object}|{first_event_start_time}|{first_event_end_time}",
                    f"{second_event_subject}|{second_event_predicate}|{second_event_object}|{second_event_start_time}|{second_event_end_time}",
                ],
                "question_level": "medium",
                "question_type": "timeline_recovery_timeline_recovery",
                "answer_type": "relation_duration",
                "temporal_relation": None,
            },
        ]
        questions += medium_type_2_questions

        if template_based:
            for question_draft in questions:
                this_type_templates = QUESTION_TEMPLATES[
                    question_draft["question_level"]
                ][question_draft["question_type"]][question_draft["answer_type"]]

                if (
                    question_draft["answer_type"] == "subject"
                    or question_draft["answer_type"] == "object"
                ):
                    """
                    Handle the Medium Type 1 Questions here: Both a and b
                    First calculate the relations, then based on relations to select the template
                    """
                    temporal_relation = self.relation_allen_time_range(
                        time_range_a=[
                            first_event_start_time_dt,
                            first_event_end_time_dt,
                        ],
                        time_range_b=[
                            second_event_start_time_dt,
                            second_event_end_time_dt,
                        ],
                    )
                    temporal_relation_semantic = temporal_relation.get("semantic")
                    question_draft["temporal_relation"] = temporal_relation["relation"]
                    random_pick_template = random.choice(
                        this_type_templates[temporal_relation_semantic]
                    )

                    question_draft["question"] = random_pick_template.format(
                        first_event_subject=first_event_subject,
                        first_event_predicate=first_event_predicate,
                        first_event_object=first_event_object,
                        temporal_relation=temporal_relation,
                        second_event_subject=second_event_subject,
                        second_event_predicate=second_event_predicate,
                        second_event_object=second_event_object,
                    )
                    # this will generate the basic temporal relation questions.
                    # TODO: we also need to generate the one duration_before, duration_after
                    # If relation is before or after, then we can generate the duration_before, duration_after
                    # Add it to variable medium_type_1_b_questions
                    if temporal_relation_semantic in ["before", "after"]:
                        random_pick_template = random.choice(
                            this_type_templates[
                                f"duration_{temporal_relation_semantic}"
                            ]
                        )
                        # get the duration year
                        # Example: 3 years before Bush as the president of US, who is the president of China?
                        # The duration is calculated based on first_end_time - second_start time
                        # NOTE: It can be extended further later to calculate first_start - second_start
                        duration = self.relation_duration_calculation(
                            time_range_a=[
                                first_event_start_time,
                                first_event_end_time,
                            ],
                            time_range_b=[
                                second_event_start_time,
                                second_event_end_time,
                            ],
                            temporal_operator=f"duration_{temporal_relation_semantic}",
                        )
                        # copy a new question draft
                        duration_question_draft = question_draft.copy(deep=True)
                        duration_question_draft["question"] = (
                            random_pick_template.format(
                                first_event_subject=first_event_subject,
                                first_event_predicate=first_event_predicate,
                                first_event_object=first_event_object,
                                temporal_relation=f"{duration} {temporal_relation}",
                                second_event_subject=second_event_subject,
                                second_event_predicate=second_event_predicate,
                                second_event_object=second_event_object,
                            )
                        )
                        duration_question_draft["temporal_relation"] = (
                            f"duration_{temporal_relation_semantic}"
                        )
                        medium_type_1_b_questions.append(duration_question_draft)
                else:
                    """
                    Handle in theory four types of questions here
                    """
                    if (
                        question_draft["answer_type"]
                        == "relation_union_or_intersection"
                    ):
                        temporal_relation = question_draft["temporal_relation"]
                        random_pick_template = random.choice(
                            this_type_templates[temporal_relation]
                        )
                        temporal_answer = self.relation_union_or_intersection(
                            time_range_a=[
                                first_event_start_time_dt,
                                first_event_end_time_dt,
                            ],
                            time_range_b=[
                                second_event_start_time_dt,
                                second_event_end_time_dt,
                            ],
                            temporal_operator=temporal_relation,
                        )

                        question_draft["question"] = random_pick_template.format(
                            first_event_subject=first_event_subject,
                            first_event_predicate=first_event_predicate,
                            first_event_object=first_event_object,
                            second_event_subject=second_event_subject,
                            second_event_predicate=second_event_predicate,
                            second_event_object=second_event_object,
                        )
                        if temporal_answer is None:
                            temporal_answer = "No Answer"
                        question_draft["answer"] = temporal_answer
                    elif question_draft["answer_type"] == "relation_allen":
                        temporal_allen_relation = self.relation_allen_time_range(
                            time_range_a=[
                                first_event_start_time_dt,
                                first_event_end_time_dt,
                            ],
                            time_range_b=[
                                second_event_start_time_dt,
                                second_event_end_time_dt,
                            ],
                        )
                        question_draft["temporal_relation"] = temporal_allen_relation[
                            "relation"
                        ]
                        # random select from [choices, true_false]
                        question_format = random.choice(["choice", "true_false"])
                        if question_format == "choice":
                            random_pick_template = random.choice(
                                this_type_templates["choice"]
                            )
                            temporal_answer = temporal_allen_relation["relation"]
                            question_draft["question"] = random_pick_template.format(
                                first_event_subject=first_event_subject,
                                first_event_predicate=first_event_predicate,
                                first_event_object=first_event_object,
                                second_event_subject=second_event_subject,
                                second_event_predicate=second_event_predicate,
                                second_event_object=second_event_object,
                            )
                            question_draft["answer"] = temporal_answer

                        else:
                            random_pick_template = random.choice(
                                this_type_templates["true_false"]
                            )
                            random_yes_no_answer = random.choice(["True", "False"])
                            if random_yes_no_answer == "True":
                                temporal_relation = temporal_allen_relation["relation"]
                            else:
                                temporal_relation = random.choice(
                                    list(
                                        set(self.allen_relations)
                                        - {temporal_allen_relation["relation"]}
                                    )
                                )
                            question_draft["question"] = random_pick_template.format(
                                first_event_subject=first_event_subject,
                                first_event_predicate=first_event_predicate,
                                first_event_object=first_event_object,
                                second_event_subject=second_event_subject,
                                second_event_predicate=second_event_predicate,
                                second_event_object=second_event_object,
                                temporal_relation=temporal_relation,
                            )
                            question_draft["answer"] = random_yes_no_answer
                    elif question_draft["answer_type"] == "relation_duration":
                        """There are four types in this category
                        - duration => which is the intersection of the two time range
                        - duration_compare => longer shorter equal
                        - sum => total duration of the two time range, which is actually the union
                        - average => average duration of the two time range
                        """
                        temporal_relation = random.choice(
                            [
                                "duration",
                                "duration_compare",
                                "sum",
                                "average",
                            ]
                        )
                        random_pick_template = random.choice(
                            this_type_templates[temporal_relation]
                        )
                        question_draft["temporal_relation"] = temporal_relation
                        if temporal_relation == "duration":
                            temporal_answer = self.relation_union_or_intersection(
                                time_range_a=[
                                    first_event_start_time_dt,
                                    first_event_end_time_dt,
                                ],
                                time_range_b=[
                                    second_event_start_time_dt,
                                    second_event_end_time_dt,
                                ],
                                temporal_operator="intersection",
                            )
                            question_draft["question"] = random_pick_template.format(
                                first_event_subject=first_event_subject,
                                first_event_predicate=first_event_predicate,
                                first_event_object=first_event_object,
                                second_event_subject=second_event_subject,
                                second_event_predicate=second_event_predicate,
                                second_event_object=second_event_object,
                            )
                        elif temporal_relation == "duration_compare":
                            temporal_relation_duration = (
                                self.relation_allen_time_duration(
                                    time_range_a=[
                                        first_event_start_time_dt,
                                        first_event_end_time_dt,
                                    ],
                                    time_range_b=[
                                        second_event_start_time_dt,
                                        second_event_end_time_dt,
                                    ],
                                )
                            )
                            temporal_answer = temporal_relation_duration["semantic"]
                            question_draft["question"] = random_pick_template.format(
                                first_event_subject=first_event_subject,
                                first_event_predicate=first_event_predicate,
                                first_event_object=first_event_object,
                                temporal_relation=temporal_answer,
                                second_event_subject=second_event_subject,
                                second_event_predicate=second_event_predicate,
                                second_event_object=second_event_object,
                            )
                        elif temporal_relation == "sum":
                            temporal_answer = self.util_average_duration_calculation(
                                time_ranges=[
                                    [
                                        first_event_start_time_dt,
                                        first_event_end_time_dt,
                                    ],
                                    [
                                        second_event_start_time_dt,
                                        second_event_end_time_dt,
                                    ],
                                ],
                                temporal_operator="sum",
                            )
                            question_draft["question"] = random_pick_template.format(
                                first_event_subject=first_event_subject,
                                first_event_predicate=first_event_predicate,
                                first_event_object=first_event_object,
                                second_event_subject=second_event_subject,
                                second_event_predicate=second_event_predicate,
                                second_event_object=second_event_object,
                            )
                        elif temporal_relation == "average":
                            temporal_answer = self.util_average_duration_calculation(
                                time_ranges=[
                                    [
                                        first_event_start_time_dt,
                                        first_event_end_time_dt,
                                    ],
                                    [
                                        second_event_start_time_dt,
                                        second_event_end_time_dt,
                                    ],
                                ],
                                temporal_operator="average",
                            )
                            question_draft["question"] = random_pick_template.format(
                                first_event_subject=first_event_subject,
                                first_event_predicate=first_event_predicate,
                                first_event_object=first_event_object,
                                second_event_subject=second_event_subject,
                                second_event_predicate=second_event_predicate,
                                second_event_object=second_event_object,
                            )
                        question_draft["answer"] = temporal_answer
                        question_draft["temporal_relation"] = temporal_relation

        # questions += medium_type_1_b_questions
        # if pharaphrased:
        #     for question_obj in questions:
        #         paraphrased_question = paraphrase_medium_question(
        #             question=question_obj["question"],
        #         )
        #         logger.info(f"paraphrased_question: {paraphrased_question}")
        #         question_obj["pharaphrased_question"] = paraphrased_question

        return questions

    def complex_question_generation(self):
        """
        This is to generate the complex question, which will involve three events and timeline

        - Type 1: Before Bush, after Kennedy, who is the president of US?
        - Type 2: Who is the first president of US among Bush, Kennedy, and Obama?
        """
        pass

    @staticmethod
    def relation_allen_time_range(
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

        logger.debug(
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
                "description": "X precedes Y",
                "category": "tr",
                "code": "tr-1",
                "semantic": "before",
            },
            (-1, -1, -1, -1, 0, -1): {
                "relation": "X m Y",
                "description": "X meets Y",
                "category": "tr",
                "code": "tr-2",
                "semantic": "meets",
            },
            (-1, -1, -1, -1, 1, -1): {
                "relation": "X o Y",
                "description": "X overlaps Y",
                "category": "tr",
                "code": "tr-3",
                "semantic": "during",
            },
            (-1, -1, -1, -1, 1, 0): {
                "relation": "X fi Y",
                "description": "X is finished by Y",
                "category": "tr",
                "code": "tr-4",
                "semantic": "finishedby",
            },
            (-1, -1, -1, -1, 1, 1): {
                "relation": "X di Y",
                "description": "X contains Y",
                "category": "tr",
                "code": "tr-5",
                "semantic": "during",
            },
            (-1, -1, 0, -1, 1, -1): {
                "relation": "X s Y",
                "description": "X starts Y",
                "category": "tr",
                "code": "tr-6",
                "semantic": "starts",
            },
            (-1, -1, 0, -1, 1, 0): {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "tr",
                "code": "tr-7",
                "semantic": "equal",
            },
            (-1, -1, 0, -1, 1, 1): {
                "relation": "X si Y",
                "description": "X is started by Y",
                "category": "tr",
                "code": "tr-8",
                "semantic": "startedby",
            },
            (-1, -1, 1, -1, 1, -1): {
                "relation": "X d Y",
                "description": "X during Y",
                "category": "tr",
                "code": "tr-9",
                "semantic": "during",
            },
            (-1, -1, 1, -1, 1, 0): {
                "relation": "X f Y",
                "description": "X finishes Y",
                "category": "tr",
                "code": "tr-10",
                "semantic": "finishes",
            },
            (-1, -1, 1, -1, 1, 1): {
                "relation": "X oi Y",
                "description": "X is overlapped by Y",
                "category": "tr",
                "code": "tr-11",
                "semantic": "during",
            },
            (-1, -1, 1, 0, 1, 1): {
                "relation": "X mi Y",
                "description": "X is met by Y",
                "category": "tr",
                "code": "tr-12",
                "semantic": "metby",
            },
            (-1, -1, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is preceded by Y",
                "category": "tr",
                "code": "tr-13",
                "semantic": "after",
            },
            (0, -1, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tp&tr",
                "code": "tptr-14",
                "semantic": "before",
            },
            (0, -1, 0, -1, 0, -1): {
                "relation": "X s Y",
                "description": "X starts Y",
                "category": "tp&tr",
                "code": "tptr-15",
                "semantic": "starts",
            },
            (0, -1, 1, -1, 1, -1): {
                "relation": "X d Y",
                "description": "X during Y",
                "category": "tp&tr",
                "code": "tptr-16",
                "semantic": "during",
            },
            (0, -1, 1, 0, 1, 0): {
                "relation": "X f Y",
                "description": "X finishes Y",
                "category": "tp&tr",
                "code": "tptr-17",
                "semantic": "finishes",
            },
            (0, -1, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tp&tr",
                "code": "tptr-18",
                "semantic": "after",
            },
            (-1, 0, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tr&tp",
                "code": "trtp-19",
                "semantic": "before",
            },
            (-1, 0, -1, -1, 0, 0): {
                "relation": "X fi Y",
                "description": "X finishes Y",
                "category": "tr&tp",
                "code": "trtp-20",
                "semantic": "finishes",
            },
            (-1, 0, -1, -1, 1, 1): {
                "relation": "X di Y",
                "description": "X during Y",
                "category": "tr&tp",
                "code": "trtp-21",
                "semantic": "during",
            },
            (-1, 0, 0, 0, 1, 1): {
                "relation": "X si Y",
                "description": "X starts Y",
                "category": "tr&tp",
                "code": "trtp-22",
                "semantic": "starts",
            },
            (-1, 0, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tr&tp",
                "code": "trtp-23",
                "semantic": "after",
            },
            (0, 0, -1, -1, -1, -1): {
                "relation": "X < Y",
                "description": "X is before Y",
                "category": "tp",
                "code": "tp-24",
                "semantic": "before",
            },
            (0, 0, 0, 0, 0, 0): {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "tp",
                "code": "tp-25",
                "semantic": "equal",
            },
            (0, 0, 1, 1, 1, 1): {
                "relation": "X > Y",
                "description": "X is after Y",
                "category": "tp",
                "code": "tp-26",
                "semantic": "after",
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
        logger.debug(f"allen_operator: {allen_operator}")
        logger.debug(f"ALLEN_OPERATOR_DICT: {ALLEN_OPERATOR_DICT[allen_operator]}")
        return ALLEN_OPERATOR_DICT[allen_operator]

    @staticmethod
    def relation_allen_time_duration(
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
                "semantic": "shorter",
            }
        elif duration_a == duration_b:
            return {
                "relation": "X = Y",
                "description": "X equals Y",
                "category": "td",
                "code": "td-2",
                "semantic": "equals",
            }
        else:
            return {
                "relation": "X > Y",
                "description": "X is longer Y",
                "category": "td",
                "code": "td-3",
                "semantic": "longer",
            }

    @staticmethod
    def relation_union_or_intersection(
        time_range_a: list[datetime, datetime],
        time_range_b: list[datetime, datetime],
        temporal_operator: str = None,
    ) -> str:
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
        ]:
            raise ValueError(
                "temporal_operator should be one of the following: intersection, union, complement"
            )
        start_a, end_a = time_range_a
        start_b, end_b = time_range_b

        if temporal_operator == "intersection":
            start = max(start_a, start_b)
            end = min(end_a, end_b)
            if start < end:
                result = (start, end)
            else:
                return None
        elif temporal_operator == "union":
            if start_a > end_b or start_b > end_a:
                return None  # No overlap, no union # ALERT
            start = min(start_a, start_b)
            end = max(end_a, end_b)
            result = (start, end)

        result_str = f"({result[0]}, {result[1]})"

        return result_str

    @staticmethod
    def relation_ordinal_time_range(
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
    def relation_duration(
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

    @staticmethod
    def relation_duration_calculation(
        time_range_a: list[datetime, datetime],
        time_range_b: list[datetime, datetime],
        temporal_operator: str = None,
    ) -> timedelta:
        """
        We will calculate the time difference between two time ranges

        However, there are several combination we can do here

        - duration_before => abs(time_range_a[1] - time_range_b[0])
        - duration_after => abs(time_range_b[1] - time_range_a[0])
        We also have other combinations, but we will not consider them here

        Args:
            time_range_a (list[datetime, datetime]): The first time range
            time_range_b (list[datetime, datetime]): The second time range
            temporal_operator (str): The temporal operator

        Returns:
            timedelta: The time difference between two time ranges

        """
        if temporal_operator is None or temporal_operator not in [
            "duration_before",
            "duration_after",
        ]:
            raise ValueError(
                "temporal_operator should be one of the following: duration_before, duration_after"
            )
        if temporal_operator == "duration_before":
            return abs(time_range_a[1] - time_range_b[0])
        if temporal_operator == "duration_after":
            return abs(time_range_b[1] - time_range_a[0])
        return None

    @staticmethod
    def util_str_to_datetime(
        time_range_a: list[str, str], time_range_b: list[str, str]
    ):
        """
        Convert the string to datetime

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
        return start_time1, end_time1, start_time2, end_time2

    def util_average_duration_calculation(
        self, time_ranges: list[[datetime, datetime]], temporal_operator: str = None
    ):
        try:
            if temporal_operator == "average":
                durations = [
                    abs(time_range[1] - time_range[0]) for time_range in time_ranges
                ]
                average_d = sum(durations) / len(durations)
                return self.utils_format_np_datetime(average_d)

            if temporal_operator == "sum":
                durations = [
                    abs(time_range[1] - time_range[0]) for time_range in time_ranges
                ]
                total = sum(durations)
                return self.utils_format_np_datetime(total)
            return None
        except Exception as e:
            logger.error(f"Error in util_average_duration_calculation: {e}")
            return None

    @staticmethod
    def utils_format_np_datetime(np_date_delta):
        td = timedelta(seconds=np_date_delta / (np.timedelta64(1, "s")))
        days = td.days
        seconds = td.seconds
        microseconds = td.microseconds

        # Compute hours, minutes, and remaining seconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # also get how many years
        years = days // 365
        if years > 2000:
            return "forever"
        months = (days % 365) // 30
        days = (days % 365) % 30
        human_readable_format = f"{years} years, {months} months, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
        return human_readable_format

    @property
    def allen_relations(self):
        return [
            "X < Y",
            "X m Y",
            "X o Y",
            "X fi Y",
            "X di Y",
            "X s Y",
            "X = Y",
            "X si Y",
            "X d Y",
            "X f Y",
            "X oi Y",
            "X mi Y",
            "X > Y",
        ]


if __name__ == "__main__":
    generator = TKGQAGenerator(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    """
    Question Types:
    
    - Simple: Timeline and One Event Involved
        - ? P O TS TE
        - S P ? TS TE
        - S P O ? TE
        - S P O TS ?
        - S P O ? ?
        - S P O Duration?
    
    - Medium: Timeline and Two Events Involved
        - Timeline Recover => Temporal Constrainted Retrieval
        - Timeline Recover + Timeline Recover
    """

    # this will include all retrieval type questions include timestamp/duration
    # generator.simple_question_generation()
    generator.medium_question_generation()
