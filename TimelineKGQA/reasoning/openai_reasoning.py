from openai import OpenAI

from TimelineKGQA.utils import get_logger

logger = get_logger(__name__)

client = OpenAI()


def reasoning_temporal_questions(prompt: str):
    """

    Args:
        prompt (str): The question to reason.
    """
    try:
        logger.info(f"Reasoning the question: {prompt}")
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert on reasoning temporal questions.
                                  Give us the answer to the following question.
                                  """,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.error(e)
        return ""
