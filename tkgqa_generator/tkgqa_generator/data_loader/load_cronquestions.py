from tkgqa_generator.constants import (
    DATA_DIR,
    DATA_ICEWS_DICTS_DATA_DIR,
    DATA_ICEWS_EVENTS_DATA_DIR,
    DB_CONNECTION_STR,
    DOC_DIR,
)
from tkgqa_generator.utils import API, get_logger, timer
import pickle
import pandas as pd

logger = get_logger(__name__)


class CronQuestions:
    def __init__(self):
        self.cron_question_dir = DATA_DIR / "CronQuestions" / "questions"

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


if __name__ == "__main__":
    with timer(logger, "Loading cron questions"):
        cron_questions = CronQuestions()
        cron_questions.load_questions()
