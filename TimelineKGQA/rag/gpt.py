import argparse
from typing import List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tiktoken
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
from tqdm import tqdm
from transformers.models.tapas.tokenization_tapas import Question

from TimelineKGQA.constants import LOGS_DIR
from TimelineKGQA.openai_utils import embedding_content
from TimelineKGQA.rag.metrics import hit_n, mean_reciprocal_rank
from TimelineKGQA.utils import get_logger, timer

logger = get_logger(__name__)


class RAGRank:
    def __init__(self, table_name, host, port, user, password, db_name="tkgqa"):
        self.engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        )
        self.table_name = table_name
        self.load_event_data()

    def load_event_data(self):
        with timer(logger, "Load Event Data"):
            self.event_df = pd.read_sql(
                f"SELECT * FROM {self.table_name};", self.engine
            )
            self._process_embeddings(
                ["embedding", "subject_embedding", "object_embedding"]
            )

    def _process_embeddings(self, columns):
        for col in columns:
            if col in self.event_df.columns:
                self.event_df[col] = self.event_df[col].apply(
                    lambda x: list(map(float, x[1:-1].split(",")))
                )

    def add_embedding_column(self):
        with self.engine.connect() as cursor:
            for col in [
                "embedding",
                "subject_embedding",
                "predicate_embedding",
                "object_embedding",
                "start_time_embedding",
                "end_time_embedding",
            ]:
                if not cursor.execute(
                    text(
                        f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.table_name}' AND column_name = '{col}';"
                    )
                ).fetchone():
                    cursor.execute(
                        text(f"ALTER TABLE {self.table_name} ADD COLUMN {col} vector;")
                    )
            cursor.commit()

    def embed_facts(self):
        df = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE embedding IS NULL;", self.engine
        )
        if df.empty:
            return
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding Facts"):
            content = f"{row['subject']} {row['predicate']} {row['object']} {row['start_time']} {row['end_time']}"
            embedding = embedding_content(content)
            with self.engine.connect() as cursor:
                cursor.execute(
                    text(
                        f"UPDATE {self.table_name} SET embedding = array{embedding}::vector WHERE id = {row['id']};"
                    )
                )
                cursor.commit()

    def embed_kg(self):
        df = pd.read_sql(f"SELECT * FROM {self.table_name};", self.engine)
        if df.empty:
            return
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding KG"):
            subject_embedding = embedding_content(row["subject"])
            object_embedding = embedding_content(row["object"])
            with self.engine.connect() as cursor:
                cursor.execute(
                    text(
                        f"UPDATE {self.table_name} SET subject_embedding = array{subject_embedding}::vector, object_embedding = array{object_embedding}::vector WHERE id = {row['id']};"
                    )
                )
                cursor.commit()

    def benchmark_naive_rag(self, semantic_parse: bool = False):
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
        questions_df = self._load_questions()
        similarities = self._calculate_similarities(questions_df)

        if semantic_parse:
            mask_result_matrix = self.semantic_parse(questions_df)
            similarities = similarities + mask_result_matrix

        top_30_values, top_30_indices = torch.topk(similarities, 30, dim=1)

        ranks = self._evaluate_rankings(questions_df, top_30_indices)
        self._save_and_log_results(ranks, "naive", semantic_parse)

    def benchmark_graph_rag(self, semantic_parse=False):
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
        questions_df = self._load_questions()
        similarities = self._calculate_similarities(questions_df)
        if semantic_parse:
            mask_result_matrix = self.semantic_parse(questions_df)
            similarities = similarities + mask_result_matrix

        top_30_values, top_30_indices = torch.topk(similarities, 30, dim=1)
        ranks = self._evaluate_rankings(questions_df, top_30_indices)
        self._save_and_log_results(ranks, "graph", semantic_parse)

    def _load_questions(self):
        questions_df = pd.read_sql(
            f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL;",
            self.engine,
        )
        questions_df["embedding"] = questions_df["embedding"].apply(
            lambda x: list(map(float, x[1:-1].split(",")))
        )
        return questions_df

    def _calculate_similarities(self, questions_df):
        q_emb = np.array(questions_df["embedding"].tolist(), dtype="float64")
        # s_emb = np.array(self.event_df["subject_embedding"].tolist(), dtype="float64")
        # o_emb = np.array(self.event_df["object_embedding"].tolist(), dtype="float64")
        e_emb = np.array(self.event_df["embedding"].tolist(), dtype="float64")
        # return torch.tensor(
        #     np.dot(q_emb, s_emb.T) + np.dot(q_emb, o_emb.T) + np.dot(q_emb, e_emb.T)
        # )
        return torch.tensor(np.dot(q_emb, e_emb.T))

    def _evaluate_rankings(self, questions_df, top_30_indices):
        ranks = []
        for index, row in tqdm(
            questions_df.iterrows(), total=len(questions_df), desc="Evaluating Rankings"
        ):
            top_30_events = top_30_indices[index].tolist()
            facts = self.event_df.iloc[top_30_events]["fact"].tolist()
            ids = self.event_df.iloc[top_30_events]["id"].tolist()
            relevant_facts = row["events"]
            rank = [1 if fact in relevant_facts else 0 for fact in facts]
            ranks.append(
                {
                    "question": row["question"],
                    "rank": {
                        "rank": rank,
                        "labels": {"complex": 3, "medium": 2, "simple": 1}[
                            row["question_level"]
                        ],
                    },
                    "top_30_events": ids,
                }
            )
        return pd.DataFrame(ranks)

    def _save_and_log_results(self, ranks_df, prefix, semantic_parse):
        ranks_df.to_csv(LOGS_DIR / f"{prefix}_ranks.csv")
        self.log_metrics(ranks_df, "all", semantic_parse)

    def log_metrics(self, ranks_df, question_level, semantic_parse):
        metrics = ["mrr", "hit_1", "hit_3", "hit_5", "hit_10"]
        levels = ["all", "simple", "medium", "complex"]

        for level in levels:
            filtered_ranks = (
                ranks_df["rank"].tolist()
                if level == "all"
                else [
                    item
                    for item in ranks_df["rank"].tolist()
                    if item["labels"] == {"simple": 1, "medium": 2, "complex": 3}[level]
                ]
            )

            for metric in metrics:
                value = (
                    mean_reciprocal_rank(filtered_ranks)
                    if metric == "mrr"
                    else hit_n(filtered_ranks, int(metric.split("_")[1]))
                )
                logger.info(
                    f"{metric.upper()}: {value}, Question Level: {level.capitalize()}, Semantic Parse: {semantic_parse}"
                )

    def vis_question_answer_similarity(self, pk=None):
        question_df = self._get_question_data(pk)
        if isinstance(question_df, str):
            return question_df, None
        event_info = "\n".join([event for event in question_df["events"]])

        fact_data = self._get_fact_data(question_df)
        if isinstance(fact_data, str):
            return fact_data, None
        top3_facts_str, top3_value, ground_truths_rank_and_value = (
            self._get_top_wrong_facts(question_df, fact_data)
        )
        # calculate the rank
        fig = self._create_visualization(question_df, fact_data)

        info_text = f"""Question: {question_df['question']}
Level: {question_df['question_level']}
Number of facts: {len(fact_data)}
Ground Truth facts: \n{event_info}
Ground truth facts rank and similarity: {ground_truths_rank_and_value}
-----
Top 3 facts: \n{top3_facts_str}
Top 3 simlarity: {top3_value.tolist()}
"""
        return info_text, fig

    def _get_top_wrong_facts(self, question_df: pd.DataFrame, fact_data):
        """
        We want to know the rank of the correct fact in the list of facts.
        We also want to know the top 3 facts that are wrong, if Hits@3 is 0.
        """
        logger.info(question_df)
        question_embedding_str = question_df["embedding"]
        question_embedding_array = np.fromstring(
            question_embedding_str[1:-1], sep=",", dtype="float64"
        ).reshape(1, -1)
        events_embedding_array = np.array(
            self.event_df["embedding"].tolist(), dtype="float64"
        )

        similarities = torch.mm(
            torch.tensor(question_embedding_array, dtype=torch.float32),
            torch.tensor(events_embedding_array, dtype=torch.float32).T,
        )

        # Get the top 3 ids
        top3_values, top3_indices = torch.topk(similarities, 3, dim=1)
        top3_indices = top3_indices[0].tolist()
        top3_facts_df = self.event_df.iloc[top3_indices]

        top3_facts_str = "\n".join(
            [
                f"{row['subject']} {row['predicate']} {row['object']} {row['start_time']} {row['end_time']}"
                for _, row in top3_facts_df.iterrows()
            ]
        )
        top3_ids = top3_facts_df["id"].tolist()

        # Locate the indices of ground truth facts
        ground_truth_fact_ids = [fact[-1] for fact in fact_data]
        ground_truth_fact_indices = self.event_df[
            self.event_df["id"].isin(ground_truth_fact_ids)
        ].index.tolist()
        logger.info(f"Ground truth fact indices: {ground_truth_fact_indices}")

        ground_truths_rank_and_value = []
        if ground_truth_fact_indices:
            # Get the rank of the correct fact within the similarity matrix based on the index
            for i, index in enumerate(ground_truth_fact_indices):
                logger.info(
                    f"Similarity value of the correct fact {i}: {similarities[0][index].item()}"
                )
                rank = (similarities[0] >= similarities[0][index]).sum().item()
                logger.info(f"Rank of the correct fact {i}: {rank}")
                ground_truths_rank_and_value.append(
                    (rank, similarities[0][index].item())
                )
        else:
            logger.warning("No ground truth facts found in the event_df")

        logger.info(f"Top 3 facts:\n{top3_facts_str}")

        return top3_facts_str, top3_values, ground_truths_rank_and_value

    def _get_question_data(self, pk):
        query = f"SELECT * FROM {self.table_name}_questions WHERE embedding IS NOT NULL"
        query += f" AND id = {int(pk)}" if pk else " ORDER BY RANDOM() LIMIT 1"
        question_df = pd.read_sql(query, self.engine)

        if question_df.empty:
            return "No question found or no questions with embeddings"
        return question_df.iloc[0]

    def _get_fact_data(self, question_df):
        fact_data = []
        for fact in question_df["events"]:
            try:
                logger.info(fact)
                subject, predicate, object, start_time, end_time = fact.split("|")
                fact_df = self.event_df[
                    (self.event_df["subject"] == subject)
                    & (self.event_df["predicate"] == predicate)
                    & (self.event_df["object"] == object)
                    & (self.event_df["start_time"] == start_time)
                    & (self.event_df["end_time"] == end_time)
                ]
                if not fact_df.empty:
                    fact_data.append(
                        (
                            fact,
                            fact_df["embedding"].values[0],
                            fact_df["subject_embedding"].values[0],
                            fact_df["object_embedding"].values[0],
                            # add indice of the fact
                            fact_df["id"].values[0],
                        )
                    )
            except Exception as e:
                logger.error(f"Error processing fact: {fact}, {e}")
        if not fact_data:
            return f"Question: {question_df['question']}\nLevel: {question_df['question_level']}\nNo matching facts found in the event database."
        return fact_data

    def _create_visualization(self, question_df, fact_data):
        question = question_df["question"]
        question_embedding = list(map(float, question_df["embedding"][1:-1].split(",")))

        fig = plt.figure(figsize=(24, 12 * len(fact_data)))
        gs = fig.add_gridspec(3 * len(fact_data), 1, hspace=0.4)

        plt.rcParams.update(
            {"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18}
        )

        for i, (fact, fact_emb, subj_emb, obj_emb, fact_id) in enumerate(fact_data):
            # ax_text = fig.add_subplot(gs[3 * i, 0])
            # ax_text.axis('off')
            # ax_text.text(0, 0.5, f"Fact {i + 1}: {fact.replace('|', ' | ')}", fontsize=18, wrap=True)

            ax_matrix = fig.add_subplot(gs[3 * i + 1 : 3 * i + 3, 0])
            similarities = self.calculate_similarities(
                question, question_embedding, fact_emb, subj_emb, obj_emb
            ).T
            sns.heatmap(
                similarities,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                cbar=False,
                ax=ax_matrix,
                annot_kws={"size": 14},
            )
            compoents = fact.split("|")
            subject = compoents[0]
            object_content = compoents[2]
            ax_matrix.set_yticklabels(
                ["Fact", subject, object_content], rotation=45, va="center"
            )
            ax_matrix.set_xticklabels(
                ["Question"] + self.word_tokenize(question), rotation=45, ha="right"
            )
            ax_matrix.set_title(
                f"{question} \n Similarity Matrix for {fact}", fontsize=16, pad=20
            )

            # Remove top and left spines
            ax_matrix.spines["top"].set_visible(False)
            ax_matrix.spines["right"].set_visible(False)
            ax_matrix.spines["left"].set_visible(False)

        return fig

    def calculate_similarities(
        self,
        question,
        question_embedding,
        fact_embedding,
        subj_embedding,
        obj_embedding,
    ):
        question_words = self.word_tokenize(question)
        similarities = [
            self._cosine_similarity(
                question_embedding, [fact_embedding, subj_embedding, obj_embedding]
            )
        ]
        for word in question_words:
            word_embedding = self.get_word_embedding(word)
            similarities.append(
                self._cosine_similarity(
                    word_embedding, [fact_embedding, subj_embedding, obj_embedding]
                )
            )
        return np.array(similarities)

    def _cosine_similarity(self, embedding1, embeddings2):
        return [cosine_similarity([embedding1], [emb])[0][0] for emb in embeddings2]

    def word_tokenize(self, text):
        enc = tiktoken.encoding_for_model("gpt-4")
        token_ids = enc.encode(text)
        return [enc.decode([token_id]) for token_id in token_ids]

    def get_word_embedding(self, word):
        return embedding_content(word)

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


def launch_gradio_app(rag):
    """
    Input can be a question ID or left blank for random question.
    Then it should pull out the question and its associated facts.
    Also the rank of the associated facts with naive similarity.
    Also pull out all facts better than the associated facts.

    """
    iface = gr.Interface(
        fn=rag.vis_question_answer_similarity,
        inputs=gr.Textbox(label="Enter Question ID (leave blank for random)"),
        outputs=[gr.Textbox(label="Question Info"), gr.Plot()],
        title="Question-Answer Similarity Visualization",
        description="Visualize the similarity between a question and its associated facts.",
        allow_flagging="never",
    )
    iface.launch()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RAG Rank")
    parser.add_argument("--table_name", type=str, default="unified_kg_icews_actor")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--user", type=str, default="tkgqa")
    parser.add_argument("--password", type=str, default="tkgqa")
    parser.add_argument("--db_name", type=str, default="tkgqa")
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--benchmark", type=str, default="naive")
    parser.add_argument("--semantic_parse", action="store_true", default=False)
    args = parser.parse_args()

    rag = RAGRank(
        table_name=args.table_name,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        db_name=args.db_name,
    )

    if args.preprocess:
        with timer(logger, "Add Embedding Column"):
            rag.add_embedding_column()

        with timer(logger, "Embed Facts"):
            rag.embed_facts()

        with timer(logger, "Embed KG"):
            rag.embed_kg()

    if args.benchmark == "naive":
        rag.benchmark_naive_rag(semantic_parse=args.semantic_parse)

    elif args.benchmark == "graph":
        rag.benchmark_graph_rag(semantic_parse=args.semantic_parse)
    else:
        # then load page
        launch_gradio_app(rag)
