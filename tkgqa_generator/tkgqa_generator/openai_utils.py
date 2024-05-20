from openai import OpenAI
from tkgqa_generator.utils import get_logger
import os

logger = get_logger(__name__)

client = OpenAI()


def paraphrase_question(question: str,
                        answer: str = None,
                        statement: str = None,
                        answer_type: str = None,
                        model_name: str = "gpt-4o",
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

    try:
        # Some examples include:
        # Who is affiliated with the organization during a given time.
        # Which or what's the organization's name a specific guy is affiliated to.
        # When/During/when is start time ...
        # Etc.
        # If there is a statement from beginning of time to the end of time, this will mean it is always true for the whole timeline.
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert on paraphrasing questions.
                                  Especially the temporal related questions. Only return the paraphrased question, nothing else. 
                                  The ??? is the missing part, and the main part the question is asking for,
                                  it can be someone, some organisation or some time.
                                  Use diverse ways to represent the temporal aspect of the question.
                                  """
                },
                {
                    "role": "user",
                    "content": prompt_text,
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
