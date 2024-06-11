import pandas as pd
import psycopg2
from tqdm import tqdm

from tkgqa_generator.openai_utils import embedding_content
from tkgqa_generator.utils import get_logger

logger = get_logger(__name__)


class RAGRank:
    """
    Input will be a list of the questions, and the output will be ranked facts based on the questions.

    """

    def __init__(
        self,
        table_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
        db_name: str = "tkgqa",
    ):
        """

        Args:
            table_name: The table name in the database, where the facts are stored
            host: The host of the database
            port: The port of the database
            user: The user of the database
            password: The password of the database
            db_name: The name of the database

        """
        # setup the db connection
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.db_name,
        )
        self.table_name = table_name

    def add_embedding_column(self):
        """
        Add a column to the unified KG table, which will be the embedding of the fact.

        If the embedding column already exists, then we will not add the column.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.table_name}' AND column_name = 'embedding';"
            )
            if not cursor.fetchone():
                cursor.execute(
                    f"ALTER TABLE {self.table_name} ADD COLUMN embedding vector;"
                )
            self.connection.commit()

    def embed_facts(self):
        """
        Get all the facts into the embedding, and save the embedding


        Facts will be in unifed KG format, so we go into that table, and then grab the facts, and then embed them.

        We need to add a column to the unified KG table, which will be the embedding of the fact.
        """
        # get from embedding is None into dataframe
        df = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE embedding IS NULL;", self.connection
        )
        # embed the facts
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            content = f"{row['subject']} {row['predicate']} {row['object']}"
            logger.info(content)
            embedding = embedding_content(content)
            logger.info(len(embedding))
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self.table_name} SET embedding = array{embedding}::vector WHERE id = {row['id']};",
                )
                self.connection.commit()

    def rag(self, question):
        """
        RAG is a retrieval-augmented generation model that uses a retriever to find relevant context
        and then a generator to produce the final answer.

        Return top 30 facts based on the question.

        Args:
            question: The question to rank the facts on.

        Returns:
            The ranked facts based on the question.
        """
        pass


if __name__ == "__main__":
    rag = RAGRank(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    rag.add_embedding_column()
    rag.embed_facts()
