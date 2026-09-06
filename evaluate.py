import json
import inspect
from statistics import mean

from rag import (
    load_vectorstore,
    load_bm25,
    get_reranker,
    retrieve_dense,
    retrieve_bm25,
    hybrid_retrieve,
    rerank,
)


EVAL_DATASET_PATH = "evaluation_dataset.json"

DENSE_K = 10
BM25_K = 10
HYBRID_K = 10
RERANK_K = 10


# LOAD DATASET

def load_eval_dataset():

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# HELPERS

def get_chunk_ids(results):

    return [
        item["document"].metadata.get("chunk_id")
        for item in results
    ]


# METRICS

def hit_at_k(retrieved_ids, relevant_ids, k):

    retrieved_ids = retrieved_ids[:k]

    return int(
        any(chunk_id in relevant_ids for chunk_id in retrieved_ids)
    )


def recall_at_k(retrieved_ids, relevant_ids, k):

    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    retrieved_relevant = retrieved_set & relevant_set

    return len(retrieved_relevant) / len(relevant_set)


def reciprocal_rank(retrieved_ids, relevant_ids):

    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


# EVALUATE ONE METHOD

def evaluate_method(name, retrieval_function, dataset):

    hit1_scores = []
    hit3_scores = []
    hit5_scores = []

    recall1_scores = []
    recall3_scores = []
    recall5_scores = []

    rr_scores = []

    print("\n")
    print("=" * 100)
    print(name)
    print("=" * 100)

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        results = retrieval_function(question)
        retrieved_ids = get_chunk_ids(results)

        hit1 = hit_at_k(retrieved_ids, relevant_ids, 1)
        hit3 = hit_at_k(retrieved_ids, relevant_ids, 3)
        hit5 = hit_at_k(retrieved_ids, relevant_ids, 5)

        recall1 = recall_at_k(retrieved_ids, relevant_ids, 1)
        recall3 = recall_at_k(retrieved_ids, relevant_ids, 3)
        recall5 = recall_at_k(retrieved_ids, relevant_ids, 5)

        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        hit1_scores.append(hit1)
        hit3_scores.append(hit3)
        hit5_scores.append(hit5)

        recall1_scores.append(recall1)
        recall3_scores.append(recall3)
        recall5_scores.append(recall5)

        rr_scores.append(rr)

        print(f"\n[{index}/{len(dataset)}] {question}")
        print(f"Relevant:  {relevant_ids}")
        print(f"Retrieved: {retrieved_ids[:5]}")

        print(
            f"Hit@1={hit1} | "
            f"Hit@3={hit3} | "
            f"Hit@5={hit5} | "
            f"Recall@3={recall3:.3f} | "
            f"MRR contribution={rr:.3f}"
        )

    return {
        "Hit@1": mean(hit1_scores),
        "Hit@3": mean(hit3_scores),
        "Hit@5": mean(hit5_scores),
        "Recall@1": mean(recall1_scores),
        "Recall@3": mean(recall3_scores),
        "Recall@5": mean(recall5_scores),
        "MRR": mean(rr_scores),
    }


# MAIN

def main():

    print("Loading evaluation dataset...")

    dataset = load_eval_dataset()

    print(f"Questions: {len(dataset)}")

    print("\nLoading FAISS...")
    vectorstore = load_vectorstore()

    print("Loading BM25...")
    bm25, bm25_documents = load_bm25()

    print("Loading CrossEncoder...")
    reranker = get_reranker()

    print("All components loaded.")


    # A. FAISS ONLY

    def faiss_only(query):

        return retrieve_dense(
            vectorstore,
            query,
            k=DENSE_K
        )


    # B. BM25 ONLY

    def bm25_only(query):

        return retrieve_bm25(
            bm25,
            bm25_documents,
            query,
            k=BM25_K
        )


    # C. FAISS + BM25 + RRF

    def hybrid_rrf(query):

        return hybrid_retrieve(
            vectorstore=vectorstore,
            bm25=bm25,
            bm25_documents=bm25_documents,
            query=query,
            dense_k=DENSE_K,
            bm25_k=BM25_K,
            final_k=HYBRID_K
        )


    # D. HYBRID + CROSS ENCODER

    def hybrid_reranked(query):

        hybrid_results = hybrid_rrf(query)

        rerank_parameters = inspect.signature(rerank).parameters

        if "min_score" in rerank_parameters:

            return rerank(
                query=query,
                results=hybrid_results,
                reranker=reranker,
                final_k=RERANK_K,
                min_score=-1e9
            )

        return rerank(
            query=query,
            results=hybrid_results,
            reranker=reranker,
            final_k=RERANK_K
        )


    # RUN ALL METHODS

    methods = {
        "FAISS only": faiss_only,
        "BM25 only": bm25_only,
        "FAISS + BM25 + RRF": hybrid_rrf,
        "Hybrid + CrossEncoder": hybrid_reranked,
    }

    all_metrics = {}

    for name, retrieval_function in methods.items():

        metrics = evaluate_method(
            name,
            retrieval_function,
            dataset
        )

        all_metrics[name] = metrics


    # FINAL TABLE

    print("\n\n")
    print("=" * 112)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 112)

    print(
        f"{'Method':30}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR':>10}"
    )

    print("-" * 112)

    for name, metrics in all_metrics.items():

        print(
            f"{name:30}"
            f"{metrics['Hit@1']:>10.3f}"
            f"{metrics['Hit@3']:>10.3f}"
            f"{metrics['Hit@5']:>10.3f}"
            f"{metrics['Recall@1']:>12.3f}"
            f"{metrics['Recall@3']:>12.3f}"
            f"{metrics['Recall@5']:>12.3f}"
            f"{metrics['MRR']:>10.3f}"
        )


if __name__ == "__main__":
    main()