import json

from rag import load_vectorstore, load_bm25, get_reranker, retrieve_dense, retrieve_bm25, reciprocal_rank_fusion, rerank


DATASET_PATH = "threshold_dataset.json"

DENSE_K = 10
BM25_K = 10
HYBRID_K = 10

THRESHOLDS = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]


def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    dataset = load_dataset()

    print(f"Questions: {len(dataset)}")
    print("\nLoading retrieval components...")

    vectorstore = load_vectorstore()
    bm25, bm25_documents = load_bm25()
    reranker = get_reranker()

    print("Components loaded.\n")

    scored_queries = []

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        label = item["label"]

        dense_results = retrieve_dense(
            vectorstore,
            question,
            k=DENSE_K
        )

        bm25_results = retrieve_bm25(
            bm25,
            bm25_documents,
            question,
            k=BM25_K
        )

        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            final_k=HYBRID_K
        )

        scored_results = rerank(
            query=question,
            results=hybrid_results,
            reranker=reranker,
            final_k=HYBRID_K,
            min_score=-1e9
        )

        scores = [
            result["reranker_score"]
            for result in scored_results
        ]

        max_score = max(scores) if scores else float("-inf")

        scored_queries.append({
            "question": question,
            "label": label,
            "max_score": max_score
        })

        print(
            f"[{index}/{len(dataset)}] "
            f"{label:14} | "
            f"max score = {max_score:8.4f} | "
            f"{question}"
        )

    print("\n")
    print("=" * 90)
    print("THRESHOLD EVALUATION")
    print("=" * 90)

    print(
        f"{'Threshold':>12}"
        f"{'In-domain Accept':>20}"
        f"{'OOD Reject':>16}"
        f"{'Accuracy':>14}"
    )

    print("-" * 90)

    for threshold in THRESHOLDS:

        in_total = 0
        in_accepted = 0

        out_total = 0
        out_rejected = 0

        correct = 0

        for item in scored_queries:

            predicted_in_domain = item["max_score"] >= threshold
            actual_in_domain = item["label"] == "in_domain"

            if actual_in_domain:

                in_total += 1

                if predicted_in_domain:
                    in_accepted += 1

            else:

                out_total += 1

                if not predicted_in_domain:
                    out_rejected += 1

            if predicted_in_domain == actual_in_domain:
                correct += 1

        in_accept_rate = in_accepted / in_total if in_total else 0
        out_reject_rate = out_rejected / out_total if out_total else 0
        accuracy = correct / len(scored_queries)

        print(
            f"{threshold:>12.1f}"
            f"{in_accept_rate:>20.3f}"
            f"{out_reject_rate:>16.3f}"
            f"{accuracy:>14.3f}"
        )

    print("\n")
    print("=" * 90)
    print("INDIVIDUAL MAX SCORES")
    print("=" * 90)

    in_domain = [
        item
        for item in scored_queries
        if item["label"] == "in_domain"
    ]

    out_domain = [
        item
        for item in scored_queries
        if item["label"] == "out_of_domain"
    ]

    print("\nIN-DOMAIN:")

    for item in sorted(in_domain, key=lambda x: x["max_score"]):
        print(f"{item['max_score']:>8.4f} | {item['question']}")

    print("\nOUT-OF-DOMAIN:")

    for item in sorted(
        out_domain,
        key=lambda x: x["max_score"],
        reverse=True
    ):
        print(f"{item['max_score']:>8.4f} | {item['question']}")


if __name__ == "__main__":
    main()