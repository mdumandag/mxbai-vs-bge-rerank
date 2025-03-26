import os.path
import pathlib

import torch
import json
import sqlite3
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def main():
    if os.path.exists("nfcorpus_similar_texts.sqlite"):
        print("Similar texts file exists, not doing anything")
        return

    print("Preparing the similar texts file")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    corpus_dataset = load_dataset("BeIR/nfcorpus", "corpus")
    corpus_texts = corpus_dataset["corpus"]["text"]
    corpus_embeddings = model.encode(
        corpus_texts, batch_size=32, convert_to_tensor=True
    )

    queries_dataset = load_dataset("BeIR/nfcorpus", "queries")
    queries = queries_dataset["queries"]["text"]
    query_embeddings = model.encode(queries, batch_size=32, convert_to_tensor=True)

    conn = sqlite3.connect("nfcorpus_similar_texts.sqlite")
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE query_similar_texts (
        query_text TEXT,
        similar_texts JSONB
    )
    """
    )

    batch_size = 10
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]
        batch_query_embeddings = query_embeddings[i : i + batch_size]

        batch_similarities = torch.nn.functional.cosine_similarity(
            batch_query_embeddings.unsqueeze(1), corpus_embeddings.unsqueeze(0), dim=-1
        )

        for query, query_sims in zip(batch_queries, batch_similarities):
            top_indices = torch.topk(query_sims, k=min(1000, len(query_sims)))[1]
            similar_texts = [corpus_texts[idx] for idx in top_indices]

            cursor.execute(
                """
            INSERT INTO query_similar_texts (query_text, similar_texts) 
            VALUES (?, ?)
            """,
                (query, json.dumps(similar_texts)),
            )

        conn.commit()
        print(f"Processed batch {i // batch_size + 1}")

    conn.close()


if __name__ == "__main__":
    main()
