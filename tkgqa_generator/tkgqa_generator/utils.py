import logging
import sys
from datetime import datetime
from typing import Optional

import requests


def get_logger(name):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler and set level to debug
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatter, start with file name and line of code (line number) that issued the log statement
    formatter = logging.Formatter(
        "%(asctime)s|%(filename)s|Line: %(lineno)d -- %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)
    return logger


class API:
    def __init__(self, domain: str = "https://api.nlp-tlp.org", token: str = ""):
        self.domain = domain
        self.token = token

    def create_embedding(self, text: str, model_name: str = "Mixtral-8x7b"):
        url = f"{self.domain}/llm/call-llm/create-embedding/"
        r = requests.post(
            url,
            data={"model_name": model_name, "prompt": text},
            headers={"Authorization": f"Token {self.token}"},
        )
        return r.json()
