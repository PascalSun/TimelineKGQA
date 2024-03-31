import logging
import sys
import time
from datetime import datetime
from logging import Logger
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


logger = get_logger(__name__)


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
        logger.info(f"status code: {r.status_code}")
        logger.info(r.text)
        return r.json()

    def queue_create_embedding(
        self, prompts: str, model_name: str = "Mixtral-8x7b", name: str = "icews_actor"
    ):
        url = f"{self.domain}/queue_task/llm_batch/"
        r = requests.post(
            url,
            data={
                "model_name": model_name,
                "name": name,
                "prompts": prompts,
                "llm_task_type": "create_embedding",
                "task_worker": "gpu",
            },
            headers={"Authorization": f"Token {self.token}"},
        )
        return r.json()


class timer:
    """
    util function used to log the time taken by a part of program
    """

    def __init__(self, logger: Logger, message: str):
        """
        init the timer

        Parameters
        ----------
        logger: Logger
            logger to write the logs
        message: str
            message to log, like start xxx
        """
        self.message = message
        self.logger = logger
        self.start = 0
        self.duration = 0
        self.sub_timers = []

    def __enter__(self):
        """
        context enter to start write this
        """
        self.start = time.time()
        self.logger.info("Starting %s" % self.message)
        return self

    def __exit__(self, context, value, traceback):
        """
        context exit will write this
        """
        self.duration = time.time() - self.start
        self.logger.info(f"Finished {self.message}, that took {self.duration:.3f}")
