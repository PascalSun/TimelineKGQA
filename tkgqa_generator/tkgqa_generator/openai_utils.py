from openai import OpenAI
from tkgqa_generator.utils import get_logger
import os

logger = get_logger(__name__)

client = OpenAI()


def paraphrase_question(question: str,
                        answer: str = None,
                        model_name: str = "gpt-3.5-turbo",
                        ) -> str:
    """
    Paraphrases the given question using the OpenAI model specified.

    Args:
        question (str): The question to paraphrase.
        answer (str, optional): The answer to the question, which can help in generating a context-aware paraphrase.
        model_name (str): The model to use for paraphrasing.

    Returns:
        str: The paraphrased question.
    """
    prompt_text = f"Paraphrase the following question: '{question}'"
    if answer:
        prompt_text += f" Answer: '{answer}'"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert on paraphrasing questions.
                                  Especially the temporal related questions. Only return the paraphrased question, nothing else. 
                                  General domain of the questions and statements are: Someone affiliated with some organization during some time."""
                },
                {
                    "role": "user",
                    "content": f"""The raw statement with ? is '{question}'.
                                    The answer to the question is '{answer}'.
                                    Please paraphrase the question into natural language properly (which means use the proper who, where, when, etc).
                                    """,
                }
            ],
            max_tokens=100,
            temperature=0.3,
            stop=["\n"]
        )
        paraphrased_question = response.choices[0].message.content
        return paraphrased_question
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
