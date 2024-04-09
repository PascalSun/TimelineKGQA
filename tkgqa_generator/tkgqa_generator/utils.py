import logging
import sys
import time
from logging import Logger
from typing import List
import requests


def get_logger(name):
    # Create a logger
    the_logger = logging.getLogger(name)
    the_logger.setLevel(logging.INFO)

    # Create console handler and set level to debug
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatter, start with file name and line of code (line number) that issued the log statement
    formatter = logging.Formatter(
        "%(asctime)s|%(filename)s|Line: %(lineno)d -- %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to the_logger
    the_logger.addHandler(console_handler)
    return the_logger


logger = get_logger(__name__)


class API:
    def __init__(self, domain: str = "https://api.nlp-tlp.org", token: str = ""):
        self.domain = domain
        self.token = token

    def queue_create_embedding(
            self, prompts: List[str], model_name: str = "Mixtral-8x7b", name: str = "icews_actor"
    ):
        url = f"{self.domain}/queue_task/llm_batch/"
        r = requests.post(
            url,
            data={
                "model_name": model_name,
                "name": name,
                "prompts": prompts,
                "llm_task_type": "create_embedding",
                "task_type": "gpu",
            },
            headers={"Authorization": f"Token {self.token}"},
        )
        return r.json()

    def get_task_status(self, task_id: str):
        url = f"{self.domain}/queue_task/{task_id}/status"
        r = requests.get(url, headers={"Authorization": f"Token {self.token}"})
        return r.json()

    def queue_embedding_and_wait_for_result(
            self, prompts: List[str], model_name: str = "Mixtral-8x7b", name: str = "icews_actor"
    ):
        res_json = self.queue_create_embedding(prompts, model_name, name)
        task_id = res_json["task_ids"][0]
        logger.info(f"Task ID: {task_id}")
        while True:
            res_json = self.get_task_status(task_id)
            if res_json["status"] == "completed":
                desc = res_json["description"]
                return eval(desc)
            time.sleep(1)


class timer:
    """
    util function used to log the time taken by a part of program
    """

    def __init__(self, the_logger: Logger, message: str):
        """
        init the timer

        Parameters
        ----------
        the_logger: Logger
            logger to write the logs
        message: str
            message to log, like start xxx
        """
        self.message = message
        self.logger = the_logger
        self.start = 0
        self.duration = 0
        self.sub_timers = []

    def __enter__(self):
        """
        context enters to start to write this
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
