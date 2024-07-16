import json

import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm


class FinetuneLLM:
    """
    Fine tune the LLM with the train QA pairs

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
        table_name: Name of the table
        host: Host name
        port: Port number
        user: User name
        password: Password
        db_name: Database name

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
        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions", self.engine
        )
        fine_tune_data = []
        for index, row in tqdm(
            questions_df.iterrows(),
            total=questions_df.shape[0],
            desc="Generating finetune data",
        ):
            question = row["question"]
            answer = row["answer"]

            """           
            {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
            {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
            {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}
            """
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
            if index > 10:
                break
        # dump it into a jsonl file
        with open("finetune_data.jsonl", "w") as f:
            for line in fine_tune_data:
                f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    fine_tune_llm = FinetuneLLM(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    fine_tune_llm.generate_finetune_data()
