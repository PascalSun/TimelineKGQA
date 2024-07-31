import json
import time

import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine, text
from tqdm import tqdm

from TimelineKGQA.constants import LOGS_DIR
from TimelineKGQA.utils import get_logger

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

    Comparison of experiments:

    - Fine tuned QA with paraphrased questions
    - Fine tuned QA, answer as question
    - Fine tuned with simple QA, and ask the relevant medium question
    """

    def __init__(
        self,
        table_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
        db_name: str,
        fine_tune_model: str = "gpt-3.5-turbo-1106",
        paraphrased_model: str = "gpt-3.5-turbo-1106",
    ):
        """
        Args:
            table_name (str): The table name
            host (str): The host
            port (int): The port
            user (str): The user
            password (str): The password
            db_name (str): The db name
            fine_tune_model (str): The fine tune model
            paraphrased_model (str): The paraphrased model


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

        self.fine_tune_model = fine_tune_model
        self.paraphrased_model = paraphrased_model

    def generate_finetune_data_paraphrased_questions(
        self, number_of_questions: int = 10, identifier_file_name: str = "paraphrased"
    ):
        """
        Args:
            number_of_questions (int): The number of questions
            identifier_file_name (str): The identifier file name

        """
        paraphrased_finetune_query = f"""
        SELECT * FROM unified_kg_icews_actor_questions
        WHERE question_type = 'timeline_recovery'
              or question_type = 'temporal_constrainted_retrieval'
        ORDER BY events
        LIMIT {number_of_questions} * 3;
        """

        questions_df = pd.read_sql(paraphrased_finetune_query, self.engine)
        fine_tune_data = []
        evaluation_data = []

        for index, row in tqdm(
            questions_df.iterrows(),
            total=questions_df.shape[0],
            desc="Generating finetune data",
        ):
            question = row["question"]
            answer = row["answer"]

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

            # paraphrased question
            paraphrased_question = self.paraphrased_question(question)
            if paraphrased_question:
                evaluation_data.append(
                    {
                        "question": question,
                        "paraphrased_question": paraphrased_question,
                        "answer": answer,
                    }
                )

        # dump it into a jsonl file
        with open(
            FINE_TUNE_LOGS_DIR / f"finetune_data_{identifier_file_name}.jsonl", "w"
        ) as f:
            for line in fine_tune_data:
                f.write(json.dumps(line) + "\n")

        with open(
            FINE_TUNE_LOGS_DIR / f"evaluation_data_{identifier_file_name}.jsonl", "w"
        ) as f:
            for line in evaluation_data:
                f.write(json.dumps(line) + "\n")

        self.create_fine_tune_task(
            train_filename=(
                FINE_TUNE_LOGS_DIR / f"finetune_data_{identifier_file_name}.jsonl"
            ).as_posix(),
            eval_filename=(
                FINE_TUNE_LOGS_DIR / f"evaluation_data_{identifier_file_name}.jsonl"
            ).as_posix(),
            identifier_file_name=identifier_file_name,
        )

    def generate_finetune_data_answer_as_question(
        self, number_of_questions: int = 10, identifier_file_name: str = "a_as_q"
    ):
        """
        Generate the finetune data

        Args:
            number_of_questions (int): The number of questions
            identifier_file_name (str): The identifier file name

        """
        simple_finetune_query = f"""
        SELECT * FROM unified_kg_icews_actor_questions
        WHERE question_type = 'timeline_recovery'
           or question_type = 'temporal_constrainted_retrieval'
        ORDER BY events
        LIMIT {number_of_questions} * 3;
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
        # dump it into a jsonl file
        with open(
            FINE_TUNE_LOGS_DIR / f"finetune_data_{identifier_file_name}.jsonl", "w"
        ) as f:
            for line in fine_tune_data:
                f.write(json.dumps(line) + "\n")

        with open(
            FINE_TUNE_LOGS_DIR / f"evaluation_data_{identifier_file_name}.jsonl", "w"
        ) as f:
            for line in evaluation_data:
                f.write(json.dumps(line) + "\n")

        self.create_fine_tune_task(
            train_filename=(
                FINE_TUNE_LOGS_DIR / f"finetune_data_{identifier_file_name}.jsonl"
            ).as_posix(),
            eval_filename=(
                FINE_TUNE_LOGS_DIR / f"evaluation_data_{identifier_file_name}.jsonl"
            ).as_posix(),
            identifier_file_name=identifier_file_name,
        )

    def generate_finetune_data_simple_vs_medium(
        self,
        number_of_questions: int = 30,
        identifier_file_name: str = "simple_vs_medium",
    ):
        """
        Args:
            number_of_questions (int): The number of questions
            identifier_file_name (str): The identifier file name

        """
        simple_vs_medium_query = f"""
        SELECT * FROM unified_kg_icews_actor_questions
        WHERE question_level = 'medium'
        ORDER BY events
        LIMIT {number_of_questions} * 3;
        """
        questions_df = pd.read_sql(
            simple_vs_medium_query,
            self.engine,
        )
        fine_tune_data = []
        evaluation_data = []
        for index, row in tqdm(
            questions_df.iterrows(),
            total=questions_df.shape[0],
            desc="Generating finetune data",
        ):
            question = row["question"]
            answer = row["answer"]

            evaluation_data.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )

            events = row["events"]
            for event in events:
                logger.info(event)
                # query the db for simple question with this event
                simple_query = (
                    """
                    SELECT * FROM unified_kg_icews_actor_questions
                    WHERE events = '{"""
                    + event.replace("'", "''")
                    + """}'
                AND question_level = 'simple'
                LIMIT 3;
                """
                )
                logger.info(simple_query)
                simple_question_df = pd.read_sql(
                    simple_query,
                    self.engine,
                )
                logger.info(simple_question_df)
                if simple_question_df.empty:
                    continue

                for _, simple_row in simple_question_df.iterrows():
                    simple_question = simple_row["question"]
                    simple_answer = simple_row["answer"]
                    fine_tune_data.append(
                        {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a QA robot expert in temporal related questions.",
                                },
                                {"role": "user", "content": simple_question},
                                {"role": "assistant", "content": simple_answer},
                            ]
                        }
                    )

        # dump it into a jsonl file
        with open(
            FINE_TUNE_LOGS_DIR / f"finetune_data_{identifier_file_name}.jsonl", "w"
        ) as f:
            for line in fine_tune_data:
                f.write(json.dumps(line) + "\n")

        with open(
            FINE_TUNE_LOGS_DIR / f"evaluation_data_{identifier_file_name}.jsonl", "w"
        ) as f:
            for line in evaluation_data:
                f.write(json.dumps(line) + "\n")

        self.create_fine_tune_task(
            train_filename=(
                FINE_TUNE_LOGS_DIR / f"finetune_data_{identifier_file_name}.jsonl"
            ).as_posix(),
            eval_filename=(
                FINE_TUNE_LOGS_DIR / f"evaluation_data_{identifier_file_name}.jsonl"
            ).as_posix(),
            identifier_file_name=identifier_file_name,
        )

    def create_fine_tune_task(
        self, train_filename: str, eval_filename: str, identifier_file_name: str
    ):
        """
        Fine tune the LLM

        Args:
            train_filename (str): The filename
            eval_filename (str): The evaluation filename
            identifier_file_name (str): The identifier, output csv file identifier
        """
        # Upload the training file
        response = client.files.create(
            file=open(train_filename, "rb"), purpose="fine-tune"
        )
        logger.info(response.id)
        # logger.info(f"Uploaded training file: {response['id']}")
        fine_tune_res = client.fine_tuning.jobs.create(
            training_file=response.id, model=self.fine_tune_model
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
            eval_filename,
            fine_tuned_model,
            output_filename=f"evaluation_results_{identifier_file_name}.csv",
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
                temperature=0.0,
            )
            logger.info(question)
            logger.info(response.choices[0].message.content)
            fine_tune_ans = response.choices[0].message.content
            evl_df.loc[index, "fine_tune_ans"] = fine_tune_ans
        evl_df.to_csv(FINE_TUNE_LOGS_DIR / output_filename, index=False)

    def paraphrased_question(self, question: str):
        """
        Args:
            question (str): The question

        Returns:
            str: The paraphrased question

        """

        try:
            prompt = f"""
            Please paraphrase the following question with the same meaning but another way to ask:
            {question}
            Return it in json format with the key "paraphrased_question"
            """
            response = client.chat.completions.create(
                model=self.paraphrased_model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            logger.info(response.choices[0].message.content)
            paraphrased_question_str = response.choices[0].message.content
            paraphrased_question = json.loads(paraphrased_question_str).get(
                "paraphrased_question", ""
            )
            return paraphrased_question
        except Exception as e:
            logger.exception(e)
            return ""


if __name__ == "__main__":
    fine_tune_llm = FinetuneLLM(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
        fine_tune_model="gpt-4o-mini-2024-07-18",
        paraphrased_model="gpt-4o-mini",
    )
    # fine_tune_llm.generate_finetune_data_answer_as_question(number_of_questions=30)
    fine_tune_llm.generate_finetune_data_paraphrased_questions(number_of_questions=30)
    # fine_tune_llm.generate_finetune_data_simple_vs_medium(number_of_questions=30)
