import concurrent.futures
from multiprocessing import cpu_count
from typing import List

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text
from tqdm import tqdm
from TimelineKGQA.constants import LOGS_DIR
from TimelineKGQA.openai_utils import embedding_content
from TimelineKGQA.rag.metrics import hit_n, mean_reciprocal_rank
from TimelineKGQA.utils import get_logger, timer
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import gradio as gr

logger = get_logger(__name__)


def launch_gradio_app(rag):
    iface = gr.Interface(
        fn=rag.vis_question_answer_similarity,
        inputs=gr.Textbox(label="Enter Question ID (leave blank for random)"),
        outputs=[gr.Textbox(label="Question Info"), gr.Plot()],
        title="Question-Answer Similarity Visualization",
        description="Visualize the similarity between a question and its associated facts.",
        allow_flagging="never"
    )
    iface.launch()


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
        # set up the db connection
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        )

        self.table_name = table_name
        with timer(logger, "Load Event Data"):
            self.event_df = pd.read_sql(
                f"SELECT * FROM {self.table_name};", self.engine
            )

            if "embedding" in self.event_df.columns:
                try:
                    self.event_df["embedding"] = self.event_df["embedding"].apply(
                        lambda x: list(map(float, x[1:-1].split(",")))
                    )
                except Exception as e:
                    logger.error(e)

            if "subject_embedding" in self.event_df.columns:
                try:
                    self.event_df["subject_embedding"] = self.event_df["subject_embedding"].apply(
                        lambda x: list(map(float, x[1:-1].split(",")))
                    )
                except Exception as e:
                    logger.error(e)

            if "object_embedding" in self.event_df.columns:
                try:
                    self.event_df["object_embedding"] = self.event_df["object_embedding"].apply(
                        lambda x: list(map(float, x[1:-1].split(",")))
                    )
                except Exception as e:
                    logger.error(e)

        self.max_workers = cpu_count()

    def add_embedding_column(self):
        """
        Add a column to the unified KG table, which will be the embedding of the fact.

        If the embedding column already exists, then we will not add the column.
        """
        # it is the sqlalchemy connection
        with self.engine.connect() as cursor:
            for embedding_column_name in [
                "embedding",
                "subject_embedding",
                "predicate_embedding",
                "object_embedding",
                "start_time_embedding",
                "end_time_embedding",
            ]:
                result = cursor.execute(
                    text(
                        f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.table_name}' "
                        f"AND column_name = '{embedding_column_name}';"
                    )
                )
                if not result.fetchone():
                    cursor.execute(
                        text(
                            f"ALTER TABLE {self.table_name} ADD COLUMN {embedding_column_name} vector;"
                        )
                    )

                cursor.commit()

        with self.engine.connect() as cursor:
            result = cursor.execute(
                text(
                    f"SELECT column_name FROM information_schema.columns WHERE "
                    f"table_name = '{self.table_name}_questions' "
                    f"AND column_name = 'embedding';"
                )
            )
            if not result.fetchone():
                cursor.execute(
                    text(
                        f"ALTER TABLE {self.table_name}_questions ADD COLUMN embedding vector;"
                    )
                )

                cursor.commit()

    def _process_fact_row(self, row):
        """
        Args:
            row: The row of the fact

        """
        content = f"{row['subject']} {row['predicate']} {row['object']} {row['start_time']} {row['end_time']}"
        embedding = embedding_content(content)
        with self.engine.connect() as cursor:
            cursor.execute(
                text(
                    f"UPDATE {self.table_name} SET embedding = array{embedding}::vector "
                    f"WHERE id = {row['id']};"
                ),
            )
            cursor.commit()

    def embed_facts(self):
        """
        Get all the facts into the embedding, and save the embedding


        Facts will be in unifed KG format, so we go into that table, and then grab the facts, and then embed them.

        We need to add a column to the unified KG table, which will be the embedding of the fact.
        """
        # get from embedding is None into dataframe
        df = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE embedding IS NULL;",
            self.engine,
        )
        # embed the facts
        # check df size, if it is empty, then return
        if df.shape[0] == 0:
            return
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
        ) as executor:
            futures = []
            for index, row in tqdm(
                    df.iterrows(), total=df.shape[0], desc="Embedding Facts"
            ):
                futures.append(executor.submit(self._process_fact_row, row))
            concurrent.futures.wait(futures)

    def _process_question_row(self, row):
        embedding = embedding_content(row["question"])
        with self.engine.connect() as cursor:
            cursor.execute(
                text(
                    f"UPDATE {self.table_name}_questions SET embedding = array{embedding}::vector "
                    f"WHERE id = {row['id']};"
                ),
            )
            cursor.commit()

    def _process_kg_row(self, row):
        """
        Process the KG row, and embed the subject and object

        """
        subject_embedding = embedding_content(row["subject"])
        object_embedding = embedding_content(row["object"])
        with self.engine.connect() as cursor:
            cursor.execute(
                text(
                    f"UPDATE {self.table_name} SET subject_embedding = array{subject_embedding}::vector, "
                    f"object_embedding = array{object_embedding}::vector WHERE id = {row['id']};"
                ),
            )
            cursor.commit()

    def embed_kg(self):
        """
        Embed the subject and object of the facts for now

        """
        df = pd.read_sql(
            f"SELECT * FROM {self.table_name};",
            self.engine,
        )
        if df.shape[0] == 0:
            return

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
        ) as executor:
            futures = []
            for index, row in tqdm(
                    df.iterrows(), total=df.shape[0], desc="Embedding KG"
            ):
                futures.append(executor.submit(self._process_kg_row, row))
            concurrent.futures.wait(futures)

    def embed_questions(self):
        """
        Get all the questions into the embedding, and save the embedding

        Args:

            random_eval (bool): Whether to do the random evaluation
        """
        # get whether the question with embedding total number = 2000, if yes, do not need to continue

        df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NULL;",
            self.engine,
        )

        # embed the facts
        # check df size, if it is empty, then return
        if df.shape[0] == 0:
            return
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
        ) as executor:
            futures = []
            for index, row in tqdm(
                    df.iterrows(), total=df.shape[0], desc="Embedding Questions"
            ):
                futures.append(executor.submit(self._process_question_row, row))
            concurrent.futures.wait(futures)

    def benchmark_naive_rag(
            self,
            question_level: str = "complex",
            random_eval: bool = False,
            semantic_parse: bool = False,
    ):
        """
        Benchmark the RAG model

        Need to first do the embedding of the questions

        2000 questions * 100000 facts = 2000000 comparisons

        For each question, select top 30 fact index.

        Grab the fact, and then compare with the actual fact.

        Args:
            question_level: The level of the question, can be complex, medium, simple
            random_eval (bool): Whether to do the random evaluation
            semantic_parse: Whether to do the semantic parse

        """

        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL;",
            self.engine,
        )

        questions_df["embedding"] = questions_df["embedding"].apply(
            lambda x: list(map(float, x[1:-1].split(",")))
        )

        questions_embedding_array = np.array(
            questions_df["embedding"].tolist(), dtype="float64"
        )

        events_embedding_array = np.array(
            self.event_df["embedding"].tolist(), dtype="float64"
        )

        # calculate the similarity between the questions and the facts
        # 2000 questions x 100000 events
        # 2000 x 100000

        logger.info(questions_embedding_array.shape)
        logger.info(events_embedding_array.shape)
        similarities = torch.mm(
            torch.tensor(questions_embedding_array),
            torch.tensor(events_embedding_array).T,
        )
        logger.info(similarities.shape)

        if semantic_parse:
            """
            Filter the facts based on the entity
            if it is candidate, mark as 1
            if it is not candidate, mark as 0
            Then use this to mask the similarity matrix
            not candidate one set as -inf
            Then do the top 30
            """
            mask_result_matrix = self.semantic_parse(questions_df)
            similarities = similarities + mask_result_matrix

        # get top 30 event index based on the similarity
        # within the similarity matrix,
        top_30_values, top_30_indices = torch.topk(similarities, 30, dim=1)

        # loop the questions, get the top 30 events, compare with the actual events
        self.event_df["fact"] = (
                self.event_df["subject"]
                + "|"
                + self.event_df["predicate"]
                + "|"
                + self.event_df["object"]
                + "|"
                + self.event_df["start_time"]
                + "|"
                + self.event_df["end_time"]
        )

        ranks = []
        labels = {
            "complex": 3,
            "medium": 2,
            "simple": 1,
        }
        for index, row in tqdm(
                questions_df.iterrows(), total=questions_df.shape[0], desc="Benchmark"
        ):
            top_30_events = top_30_indices[index].tolist()
            # get all the events from self.events_df
            facts = self.event_df.iloc[top_30_events]["fact"].tolist()
            ids = self.event_df.iloc[top_30_events]["id"].tolist()
            relevant_facts = row["events"]

            rank = [1 if fact in relevant_facts else 0 for fact in facts]
            ranks.append(
                {
                    "question": row["question"],
                    "rank": {
                        "rank": rank,
                        "labels": labels[row["question_level"]],
                    },
                    "top_30_events": ids,
                }
            )
        ranks_df = pd.DataFrame(ranks)
        ranks_df.to_csv(LOGS_DIR / f"{question_level}_ranks.csv")
        mrr = mean_reciprocal_rank(ranks_df["rank"].tolist())
        self.log_metrics(ranks_df, question_level, semantic_parse)

    def benchmark_graph_rag(
            self,
            semantic_parse: bool = False,
    ):
        """

        1. Get the embedding of question
        2. Get embedding of facts
        3. Get top k subject and object based on the similarity
        4. List these facts, and do a Hit@k match
        5. Do shortest path vote

        Args:
            semantic_parse: Whether the semantic parse is used

        """
        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL;",
            self.engine,
        )

        questions_df["embedding"] = questions_df["embedding"].apply(
            lambda x: list(map(float, x[1:-1].split(",")))
        )

        questions_embedding_array = np.array(
            questions_df["embedding"].tolist(), dtype="float64"
        )

        events_subject_embedding_array = np.array(
            self.event_df["subject_embedding"].tolist(), dtype="float64"
        )
        events_object_embedding_array = np.array(
            self.event_df["object_embedding"].tolist(), dtype="float64"
        )
        event_embedding = np.array(
            self.event_df["embedding"].tolist(), dtype="float64"
        )

        # calculate the similarity between the questions and the subject and object

        similarities_subject = torch.mm(
            torch.tensor(questions_embedding_array),
            torch.tensor(events_subject_embedding_array).T,
        )

        similarities_object = torch.mm(
            torch.tensor(questions_embedding_array),
            torch.tensor(events_object_embedding_array).T,
        )

        similarity_event = torch.mm(
            torch.tensor(questions_embedding_array),
            torch.tensor(event_embedding).T,
        )

        # add the three similarities together and then do the top 30
        similarity = similarities_subject + similarities_object + similarity_event
        top_30_values, top_30_indices = torch.topk(similarity, 30, dim=1)

        top_30_subject_values, top_30_subject_indices = torch.topk(similarities_subject, 100, dim=1)
        top_30_object_values, top_30_object_indices = torch.topk(similarities_object, 100, dim=1)
        top_30_event_values, top_30_event_indices = torch.topk(similarity_event, 100, dim=1)
        # loop the questions, get the top 30 events, compare with the actual events
        self.event_df["fact"] = (
                self.event_df["subject"]
                + "|"
                + self.event_df["predicate"]
                + "|"
                + self.event_df["object"]
                + "|"
                + self.event_df["start_time"]
                + "|"
                + self.event_df["end_time"]
        )

        ranks = []
        labels = {
            "complex": 3,
            "medium": 2,
            "simple": 1,
        }
        for index, row in tqdm(
                questions_df.iterrows(), total=questions_df.shape[0], desc="Benchmark Graph RAG"
        ):
            # top_30_subject_events = top_30_subject_indices[index].tolist()
            # top_30_object_events = top_30_object_indices[index].tolist()
            # top_30_events = top_30_event_indices[index].tolist()
            #
            # # get all events shown on both top 30 subject and object
            # top_30_merge_events = list(set(top_30_subject_events).intersection(top_30_object_events))
            # top_30_merge_events = list(set(top_30_merge_events).intersection(top_30_events))
            top_30_events = top_30_indices[index].tolist()
            # logger.info(f"Top 30 events: {top_30_events}")

            facts = self.event_df.iloc[top_30_events]["fact"].tolist()
            ids = self.event_df.iloc[top_30_events]["id"].tolist()
            relevant_facts = row["events"]
            rank = [1 if fact in relevant_facts else 0 for fact in facts]
            ranks.append(
                {
                    "question": row["question"],
                    "rank": {
                        "rank": rank,
                        "labels": labels[row["question_level"]],
                    },
                    "top_30_events": ids,
                }
            )
        ranks_df = pd.DataFrame(ranks)
        ranks_df.to_csv(LOGS_DIR / "graph_ranks.csv")

        self.log_metrics(ranks_df, "all", semantic_parse)

    def log_metrics(self, ranks_df, question_level, semantic_parse):
        """
        Args:
            ranks_df: The ranks dataframe
            question_level: The question level
            semantic_parse: Whether the semantic parse is used
        """
        logger.debug(ranks_df["rank"])
        mrr = mean_reciprocal_rank(ranks_df["rank"].tolist())
        logger.info(
            f"MRR: {mrr}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_1 = hit_n(ranks_df["rank"].tolist(), 1)
        logger.info(
            f"Hit@1: {hit_1}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_3 = hit_n(ranks_df["rank"].tolist(), 3)
        logger.info(
            f"Hit@3: {hit_3}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_5 = hit_n(ranks_df["rank"].tolist(), 5)
        logger.info(
            f"Hit@5: {hit_5}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )
        hit_10 = hit_n(ranks_df["rank"].tolist(), 10)
        logger.info(
            f"Hit@10: {hit_10}, Question Level: {question_level}, Semantic Parse: {semantic_parse}"
        )

        # summary based on the question level
        mrr_simple = mean_reciprocal_rank(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1]
        )
        logger.info(
            f"MRR: {mrr_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_1_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 1
        )
        logger.info(
            f"Hit@1: {hit_1_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_3_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 3
        )
        logger.info(
            f"Hit@3: {hit_3_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_5_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 5
        )

        logger.info(
            f"Hit@5: {hit_5_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        hit_10_simple = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 1], 10
        )

        logger.info(
            f"Hit@10: {hit_10_simple}, Question Level: Simple, Semantic Parse: {semantic_parse}"
        )

        mrr_medium = mean_reciprocal_rank(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2]
        )
        logger.info(
            f"MRR: {mrr_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_1_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 1
        )

        logger.info(
            f"Hit@1: {hit_1_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_3_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 3
        )

        logger.info(
            f"Hit@3: {hit_3_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_5_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 5
        )

        logger.info(
            f"Hit@5: {hit_5_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        hit_10_medium = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 2], 10
        )

        logger.info(
            f"Hit@10: {hit_10_medium}, Question Level: Medium, Semantic Parse: {semantic_parse}"
        )

        mrr_complex = mean_reciprocal_rank(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3]
        )

        logger.info(
            f"MRR: {mrr_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_1_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 1
        )

        logger.info(
            f"Hit@1: {hit_1_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_3_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 3
        )

        logger.info(
            f"Hit@3: {hit_3_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_5_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 5
        )

        logger.info(
            f"Hit@5: {hit_5_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

        hit_10_complex = hit_n(
            [item for item in ranks_df["rank"].tolist() if item["labels"] == 3], 10
        )

        logger.info(
            f"Hit@10: {hit_10_complex}, Question Level: Complex, Semantic Parse: {semantic_parse}"
        )

    def semantic_parse(self, questions_df: pd.DataFrame):
        """
        Filter the facts based on the entity
        if it is candidate, mark as 1
        if it is not candidate, mark as 0
        Then use this to mask the similarity matrix
        not candidate one set as -inf
        Then do the top 30

        Return will be a len(questions_df) x len(events_df) matrix, with 1 and 0
        """

        def extract_entities(events: List[str]):
            """
            Extract the entities from the event

            Args:
                events: The event string

            Returns:
                The entities in the event
            """
            the_entities = []
            for event in events:
                try:
                    elements = event.split("|")
                    the_entities.append(elements[0])
                    the_entities.append(elements[2])
                except Exception as e:
                    logger.debug(e)

            return the_entities

        questions_df["entities"] = questions_df["events"].apply(
            lambda x: extract_entities(x)
        )
        # get all value to be -2
        result_matrix = np.zeros(
            (len(questions_df), len(self.event_df)), dtype="float64"
        )
        result_matrix = result_matrix - 2
        for index, row in tqdm(
                questions_df.iterrows(), total=questions_df.shape[0], desc="Semantic Parse"
        ):
            entities = row["entities"]
            for entity in entities:
                result_matrix[index] = np.where(
                    self.event_df["subject"] == entity, 1, result_matrix[index]
                )
                result_matrix[index] = np.where(
                    self.event_df["object"] == entity, 1, result_matrix[index]
                )
        return result_matrix

    def calculate_similarities(self, question, question_embedding, fact_embedding, subj_embedding, obj_embedding):
        question_words = self.word_tokenize(question)
        similarities = []
        # Add similarity for the full question
        similarities.append([
            cosine_similarity([question_embedding], [fact_embedding])[0][0],
            cosine_similarity([question_embedding], [subj_embedding])[0][0],
            cosine_similarity([question_embedding], [obj_embedding])[0][0]
        ])
        # Add similarities for each word
        for word in question_words:
            word_embedding = self.get_word_embedding(word)
            similarities.append([
                cosine_similarity([word_embedding], [fact_embedding])[0][0],
                cosine_similarity([word_embedding], [subj_embedding])[0][0],
                cosine_similarity([word_embedding], [obj_embedding])[0][0]
            ])
        return np.array(similarities)

    def vis_question_answer_similarity(self, pk=None):
        try:
            question_df = pd.read_sql(
                f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL;",
                self.engine,
            )
            if pk:
                question_df = question_df[question_df["id"] == int(pk)]
            else:
                question_df = question_df.sample(n=1)

            if question_df.empty:
                return "No question found or no questions with embeddings", None

            question_df = question_df.iloc[0]
            question = question_df["question"]
            level = question_df["question_level"]
            question_embedding = list(map(float, question_df["embedding"][1:-1].split(",")))
            question_facts = question_df["events"]

            if not question_facts:
                return f"Question: {question}\nLevel: {level}\nNo facts found for this question.", None

            fact_data = []
            for fact in question_facts:
                subject, predicate, object, start_time, end_time = fact.split("|")
                fact_df = self.event_df[
                    (self.event_df["subject"] == subject)
                    & (self.event_df["predicate"] == predicate)
                    & (self.event_df["object"] == object)
                    & (self.event_df["start_time"] == start_time)
                    & (self.event_df["end_time"] == end_time)
                    ]
                if fact_df.empty:
                    continue
                fact_embedding = fact_df["embedding"].values[0]
                subject_embedding = fact_df["subject_embedding"].values[0]
                object_embedding = fact_df["object_embedding"].values[0]
                fact_data.append((fact, fact_embedding, subject_embedding, object_embedding))

            if not fact_data:
                return f"Question: {question}\nLevel: {level}\nNo matching facts found in the event database.", None

            # Create visualizations
            num_facts = len(fact_data)
            fig = plt.figure(figsize=(20, 10 * num_facts))
            gs = fig.add_gridspec(3 * num_facts, 1)

            for i, (fact, fact_emb, subj_emb, obj_emb) in enumerate(fact_data):
                # Text
                ax_text = fig.add_subplot(gs[3 * i, 0])
                ax_text.axis('off')
                ax_text.text(0, 0.5, f"Fact {i + 1}: {fact.replace('|', ' | ')}", fontsize=24, wrap=True)

                # Similarity matrix
                ax_matrix = fig.add_subplot(gs[3 * i + 1, 0])
                similarities = self.calculate_similarities(question, question_embedding, fact_emb, subj_emb, obj_emb)
                # transpose the similarities matrix, y axis will be the question and x axis will be the fact
                similarities = similarities.T
                sns.heatmap(similarities, annot=True, cmap='YlOrRd', cbar=False, ax=ax_matrix)

                ax_matrix.set_yticklabels(['Fact', 'Subject', 'Object'])
                ax_matrix.set_xticklabels(['Question'] + self.word_tokenize(question), rotation=0)
                ax_matrix.set_title(f"Similarity Matrix for Fact {i + 1}")

                # Blank space
                fig.add_subplot(gs[3 * i + 2, 0]).axis('off')
            # update the font size of the plot
            plt.rcParams.update({'font.size': 24})
            # plt.tight_layout()
            info_text = f"Question: {question}\nLevel: {level}\nNumber of facts: {num_facts}"
            return info_text, fig

        except Exception as e:
            logger.exception("An error occurred:")
            return f"An error occurred: {str(e)}", None

    def word_tokenize(self, text):
        # Initialize the tokenizer using OpenAI's GPT-4 tokenizer
        enc = tiktoken.encoding_for_model("gpt-4")
        # Encode the text to get the token ids
        token_ids = enc.encode(text)
        # Decode each token id back to string tokens
        tokens = [enc.decode([token_id]) for token_id in token_ids]
        return tokens

    def get_word_embedding(self, word):
        # Implement this method to get word embeddings
        # For simplicity, you can use a pre-trained model or a simple method
        # This is a placeholder implementation
        return embedding_content(word)


if __name__ == "__main__":
    metric_question_level = "all"
    rag = RAGRank(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )

    # with timer(logger, "Add Embedding Column"):
    #     rag.add_embedding_column()
    #
    # with timer(logger, "Embed Facts"):
    #     rag.embed_facts()

    # with timer(logger, "Embed KG"):
    #     rag.embed_kg()

    # with timer(logger, "Embed Questions"):
    #     rag.embed_questions()

    # with timer(logger, "Benchmark Naive RAG without semantic parse"):
    #     rag.benchmark_naive_rag(semantic_parse=False)
    #
    # with timer(logger, "Benchmark Naive RAG with semantic parse"):
    #     rag.benchmark_naive_rag(semantic_parse=True)

    # with timer(logger, "Benchmark Graph RAG without semantic parse"):
    #     rag.benchmark_graph_rag(semantic_parse=False)
    launch_gradio_app(rag)
    # rag.vis_question_answer_similarity()
