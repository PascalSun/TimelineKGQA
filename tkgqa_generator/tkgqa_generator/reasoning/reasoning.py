from tkgqa_generator.utils import get_logger, timer
import pandas as pd
from sqlalchemy import create_engine, text
from multiprocessing import cpu_count

logger = get_logger(__name__)


class TemporalReasoningEvaluator:
    def __init__(self,
                 table_name: str,
                 host: str,
                 port: int,
                 user: str,
                 password: str,
                 db_name: str = "tkgqa",
                 ):
        """

        Args:
            table_name (str): The table name to be evaluated.
            host (str): The host of the database.
            port (int): The port of the database.
            user (str): The user of the database.
            password (str): The password of the database.
            db_name (str): The database name. Default is "tkgqa".
        """

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
        self.table_name = table_name
        with timer(logger, "Loading the table"):
            self.qa_df = pd.read_sql_table(table_name, self.engine)
            logger.info(f"Loaded {len(self.qa_df)} records from the table {table_name}.")

        self.max_workers = cpu_count

    def evaluate(self):
        pass


if __name__ == "__main__":
    table_name = "unified_kg_cron_questions"
    host = "localhost"
    port = 5433
    user = "tkgqa"
    password = "tkgqa"
    db_name = "tkgqa"
    evaluator = TemporalReasoningEvaluator(table_name, host, port, user, password, db_name)
    evaluator.evaluate()
