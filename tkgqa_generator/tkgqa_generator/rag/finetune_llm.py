import json
import time

import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine, text
from tqdm import tqdm

from tkgqa_generator.constants import LOGS_DIR
from tkgqa_generator.utils import get_logger

logger = get_logger(__name__)

client = OpenAI()

FINE_TUNE_LOGS_DIR = LOGS_DIR / "finetune_llm"

FINE_TUNE_LOGS_DIR.mkdir(exist_ok=True, parents=True)


class FinetuneLLM:
    """
    Fine tune the LLM with the train QA pairs

    We can not provide a fair game between this and RAG, and text2sql.
    They all can be information retrieval task, which will be more explainable.

    And if the LLM do not have the knowledge, it will not make sense even you fine tune it.
    So what we propose should be, we ingest the information to the LLM, and ask it from other perspective.

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
            table_name (str): The table name
            host (str): The host
            port (int): The port
            user (str): The user
            password (str): The password
            db_name (str): The db name


        """
        self.table_name = table_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name

        self.engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
        )

    def generate_finetune_data(self):
        """
        Generate the finetune data

        """
        simple_finetune_query = """
        SELECT * FROM unified_kg_icews_actor_questions
        WHERE question_type = 'timeline_recovery'
           or question_type = 'temporal_constrainted_retrieval'
        ORDER BY events;
        """

        questions_df = pd.read_sql(simple_finetune_query, self.engine)
        fine_tune_data = []
        evaluation_data = []

        for index, row in tqdm(
            questions_df.iterrows(),
            total=questions_df.shape[0],
            desc="Generating finetune data",
        ):
            question = row["question"]
            answer = row["answer"]
            if row["question_type"] == "timeline_recovery":
                evaluation_data.append(
                    {
                        "question": question,
                        "answer": answer,
                    }
                )
                continue

            fine_tune_data.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a QA robot expert in temporal related questions.",
                        },
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
            if index > 40:
                break
        # dump it into a jsonl file
        with open(FINE_TUNE_LOGS_DIR / "finetune_data.jsonl", "w") as f:
            for line in fine_tune_data:
                f.write(json.dumps(line) + "\n")

        with open(FINE_TUNE_LOGS_DIR / "evaluation_data.jsonl", "w") as f:
            for line in evaluation_data:
                f.write(json.dumps(line) + "\n")

        self.create_fine_tune_task(
            train_filename=(FINE_TUNE_LOGS_DIR / "finetune_data.jsonl").as_posix(),
            eval_filename=(FINE_TUNE_LOGS_DIR / "evaluation_data.jsonl").as_posix(),
        )

    def create_fine_tune_task(self, train_filename: str, eval_filename: str):
        """
        Fine tune the LLM

        Args:
            train_filename (str): The filename
            eval_filename (str): The evaluation filename
        """
        # Upload the training file
        response = client.files.create(
            file=open(train_filename, "rb"), purpose="fine-tune"
        )
        logger.info(response.id)
        # logger.info(f"Uploaded training file: {response['id']}")
        fine_tune_res = client.fine_tuning.jobs.create(
            training_file=response.id, model="gpt-3.5-turbo-1106"
        )
        logger.info(fine_tune_res.id)

        status = "pending"
        while status not in ["succeeded", "failed"]:
            response = client.fine_tuning.jobs.retrieve(
                fine_tuning_job_id=fine_tune_res.id
            )
            status = response.status
            logger.info(f"Fine-tune job status: {status}")
            time.sleep(30)  # Wait for 30 seconds before checking the status again

        # when success, then call the model to do the QA, for further evaluation
        if status == "succeeded":
            logger.info("Fine-tuning succeeded.")
            fine_tuned_model = response.fine_tuned_model
            logger.info(f"Fine-tuned model: {fine_tuned_model}")
        else:
            logger.info("Fine-tuning failed.")
            return

        self.evaluation_fine_tune_task(
            eval_filename, fine_tuned_model, output_filename="evaluation_results.csv"
        )

    @staticmethod
    def evaluation_fine_tune_task(
        eval_filename: str, fine_tuned_model: str, output_filename: str
    ):
        # Evaluation
        evl_df = pd.read_json(eval_filename, lines=True)
        for index, row in evl_df.iterrows():
            question = row["question"]

            response = client.chat.completions.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "user", "content": question},
                ],
            )
            logger.info(question)
            logger.info(response.choices[0].message.content)
            fine_tune_ans = response.choices[0].message.content
            evl_df.loc[index, "fine_tune_ans"] = fine_tune_ans
        evl_df.to_csv(FINE_TUNE_LOGS_DIR / output_filename, index=False)


if __name__ == "__main__":
    fine_tune_llm = FinetuneLLM(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    # fine_tune_llm.generate_finetune_data()
    fine_tune_llm.evaluation_fine_tune_task(
        eval_filename=(FINE_TUNE_LOGS_DIR / "evaluation_data.jsonl").as_posix(),
        fine_tuned_model="ft:gpt-3.5-turbo-0125:ai4wa::9m2ejgUI",
        output_filename="evaluation_results.csv",
    )
