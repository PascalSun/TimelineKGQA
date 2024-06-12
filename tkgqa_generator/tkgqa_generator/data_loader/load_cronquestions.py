import pickle

import pandas as pd
from sqlalchemy import create_engine, text

from tkgqa_generator.constants import (
    DATA_DIR,
    DB_CONNECTION_STR,
)
from tkgqa_generator.utils import get_logger, timer

logger = get_logger(__name__)


class CronQuestions:
    def __init__(self):
        self.engine = create_engine(DB_CONNECTION_STR)
        self.cron_question_dir = DATA_DIR / "CronQuestions" / "questions"
        self.cron_kg_dir = DATA_DIR / "CronQuestions" / "kg"
        self.id_2_relation = None
        self.id_2_entity = None
        self.id_alias = None
        self.full_df = None

    def load_questions(self):
        # read pickle file
        question_files = [
            "test.pickle",
            "train.pickle",
            "valid.pickle",
        ]
        questions_df = pd.DataFrame()
        for question_file in question_files:
            with open(self.cron_question_dir / question_file, "rb") as f:
                questions = pickle.load(f)
                # Convert questions to a DataFrame before appending if not already in DataFrame format
                if not isinstance(questions, pd.DataFrame):
                    questions = pd.DataFrame(questions)
                questions_df = pd.concat([questions_df, questions], ignore_index=True)

        questions_df.to_csv(self.cron_question_dir / "questions.csv", index=False)
        # question,answers,answer_type,template ONLY USE this 4 columns to SQL
        questions_df = questions_df[["question", "answer_type", "template", "type"]]
        questions_df.to_sql("cron_questions", DB_CONNECTION_STR, if_exists="replace")
        logger.info(len(questions_df))

    def load_kg(self):
        """
        Load the questions to the unified KG table.
        :return:
        """
        # go to database to check whether we have a table cron_kg exists and with record, if not
        # load it from the file
        # first check whether the table exists
        load_full_kg = False
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'cron_kg'"
                )
            )
            cron_kg_table_count = (
                result.scalar()
            )  # Using scalar() to get the first column of the first row
            logger.info(cron_kg_table_count)
            if cron_kg_table_count != 0:
                result = conn.execute(text("SELECT COUNT(*) FROM cron_kg"))
                cron_kg_record_count = (
                    result.scalar()
                )  # Using scalar() to get the first column of the first row

                if cron_kg_record_count == 0:
                    load_full_kg = True
            else:
                load_full_kg = True

        if load_full_kg is False:
            with timer(logger, "read full.csv"):
                self.full_df = pd.read_sql("SELECT * FROM cron_kg", self.engine)
            return

        with timer(logger, "load basic dataset information"):
            id_2_entity = pd.read_table(
                self.cron_kg_dir / "wd_id2entity_text.txt", header=None
            )
            id_2_relation = pd.read_table(
                self.cron_kg_dir / "wd_id2relation_text.txt", header=None
            )
            # rename columns to ['id', 'entity']
            id_2_entity = id_2_entity.rename(columns={0: "id", 1: "entity"})
            # rename columns to ['id', 'relation']
            id_2_relation = id_2_relation.rename(columns={0: "id", 1: "relation"})

            # Ensure that 'id' is the index
            id_2_entity.set_index("id", inplace=True)
            id_2_relation.set_index("id", inplace=True)

            self.id_2_relation = id_2_relation
            self.id_2_entity = id_2_entity
            self.id_alias = pd.read_pickle(self.cron_kg_dir / "wd_id_to_aliases.pickle")

        with timer(logger, "load full.csv"):

            full_df = pd.read_table(self.cron_kg_dir / "full.txt", header=None)
            full_df = full_df.rename(
                columns={
                    0: "head",
                    1: "relation",
                    2: "tail",
                    3: "start_year",
                    4: "end_year",
                }
            )

            # Create mapping series
            entity_mapping = id_2_entity["entity"]
            relation_mapping = id_2_relation["relation"]

            # Convert to nlp triple format
            full_df["head"] = full_df["head"].map(entity_mapping)
            full_df["relation"] = full_df["relation"].map(relation_mapping)
            full_df["tail"] = full_df["tail"].map(entity_mapping)
            self.full_df = full_df
            # add -01-01 to start_year and end_year
            self.full_df["start_year"] = (
                self.full_df["start_year"].astype(str) + "-01-01"
            )
            self.full_df["end_year"] = self.full_df["end_year"].astype(str) + "-01-01"
            self.full_df.to_sql("cron_kg", DB_CONNECTION_STR, if_exists="replace")

    def unified_kg(self):
        """
        Get the cron_kg into the unified kg format

        Which include the following columns:
        subject
        predicate
        object
        subject_json (json) empty here
        object_json (json) empty here
        start_time (str)
        end_time (str)

        """
        cursor = self.engine.connect()
        # run sql directly from the cron_kg
        cursor.execute(
            text(
                """
                DO
                $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'unified_kg_cron') THEN
                        CREATE TABLE unified_kg_cron(
                            id SERIAL PRIMARY KEY,
                            subject TEXT,
                            subject_json JSON DEFAULT '{}'::JSON,
                            predicate TEXT,
                            predicate_json JSON DEFAULT '{}'::JSON,
                            object TEXT,
                            object_json JSON DEFAULT '{}'::JSON,
                            start_time TEXT,
                            end_time TEXT
                        );
                    END IF;
                    TRUNCATE TABLE unified_kg_cron;
                    INSERT INTO unified_kg_cron(subject, predicate, object, start_time, end_time)
                    SELECT head, relation, tail, start_year, end_year FROM cron_kg;
                END
                $$
                """
            )
        )
        cursor.commit()
        cursor.close()


if __name__ == "__main__":
    with timer(logger, "Loading cron questions"):
        cron_questions = CronQuestions()
        cron_questions.load_questions()
        cron_questions.load_kg()
        cron_questions.unified_kg()
