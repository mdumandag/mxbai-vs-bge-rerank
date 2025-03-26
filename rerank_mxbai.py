import json
import os
import sqlite3
import time

from mxbai_rerank import MxbaiRerankV2
from hdrh.histogram import HdrHistogram

TOP_K = int(os.environ.get("TOP_K", 10))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 1024))


def main():
    reranker = MxbaiRerankV2(
        model_name_or_path="mixedbread-ai/mxbai-rerank-base-v2",
        device="cuda",
        max_length=MAX_LENGTH,
        torch_dtype="float16",
    )

    reranker.rank(
        query="warm",
        documents=["up"],
        top_k=TOP_K,
        return_documents=False,
        batch_size=BATCH_SIZE,
        show_progress=False,
    )

    histogram = HdrHistogram(1, 60 * 60 * 1000, 3)

    conn = sqlite3.connect("nfcorpus_similar_texts.sqlite")
    cursor = conn.cursor()

    cursor.execute("SELECT query_text, similar_texts FROM query_similar_texts")

    for i, (query_text, similar_texts_raw) in enumerate(cursor):
        if i % 100 == 0:
            print("At query", i)

        similar_texts = json.loads(similar_texts_raw)[:TOP_K]

        start = time.perf_counter_ns()
        reranker.rank(
            query=query_text,
            documents=similar_texts,
            top_k=TOP_K,
            return_documents=False,
            batch_size=BATCH_SIZE,
            show_progress=False,
        )
        elapsed = time.perf_counter_ns() - start
        elapsed_ms = elapsed // 1_000_000
        histogram.record_value(elapsed_ms)

    conn.close()

    print("Min:", histogram.get_min_value())
    print("Max:", histogram.get_max_value())
    print("Mean:", histogram.get_mean_value())
    print("p50:", histogram.get_value_at_percentile(50))
    print("p90:", histogram.get_value_at_percentile(90))
    print("p99:", histogram.get_value_at_percentile(99))
    print("p99.9:", histogram.get_value_at_percentile(99.9))
    print("p99.99:", histogram.get_value_at_percentile(99.99))


if __name__ == "__main__":
    main()
