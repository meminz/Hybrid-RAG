import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def print_per_query(results: list[dict]) -> None:
    for i, q in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: \"{q['query']}\"")
        print(f"{'='*80}")
        print(f"  Description:      {q['description']}")
        print(f"  Expected version: {q['expected_version']}")
        print(f"  Relevant docs:    {q['relevant_docs']}")
        print()

        # Metrics table
        methods = ["hybrid", "bm25", "embedding"]
        metric_names = ["precision@1", "recall@1", "ndcg@1",
                        "precision@2", "recall@2", "ndcg@2",
                        "precision@3", "recall@3", "ndcg@3", "mrr"]

        header = f"  {'Metric':<15}" + "".join(f"{m:>12}" for m in methods)
        print(header)
        print(f"  {'-'*15}  {'-'*12*3}")

        for metric in metric_names:
            row = f"  {metric:<15}"
            for method in methods:
                val = q["metrics"][method].get(metric, 0.0)
                row += f"{val:>12.4f}"
            print(row)

        # Ranking highlights (top 3)
        for method in methods:
            ranking_key = f"{method}_ranking"
            relevant_key = f"{method}_retrieved_relevant"
            ranking = q.get(ranking_key, [])
            relevant = q.get(relevant_key, [])

            print(f"\n  {method.upper()} top-3:")
            for rank, doc_id in enumerate(ranking[:3], 1):
                marker = " [RELEVANT]" if any(
                    r["doc_id"] == doc_id for r in relevant
                ) else ""
                print(f"    {rank}. {doc_id}{marker}")

            if relevant:
                print(f"  Retrieved relevant ({len(relevant)}):")
                for r in relevant:
                    print(f"    - {r['doc_id']} (grade={r['grade']}, rank={r['rank']})")


def print_aggregate(aggregate: dict) -> None:
    print(f"\n\n{'#'*80}")
    print(f"# AGGREGATE RESULTS ({aggregate['num_queries']} queries)")
    print(f"{'#'*80}")

    methods = ["hybrid", "bm25", "embedding"]
    metric_names = ["precision@1", "recall@1", "ndcg@1",
                    "precision@2", "recall@2", "ndcg@2",
                    "precision@3", "recall@3", "ndcg@3", "mrr"]

    # Per-method averages
    print("\n  Per-method averages:")
    header = f"  {'Metric':<20}" + "".join(f"{m:>12}" for m in methods)
    print(header)
    print(f"  {'-'*20}  {'-'*12*3}")

    for metric in metric_names:
        row = f"  {metric:<20}"
        for method in methods:
            key = f"avg_{metric}"
            val = aggregate["methods"][method].get(key, 0.0)
            row += f"{val:>12.4f}"
        print(row)



def main(filepath) -> None:
    results_path = filepath

    data = load_results(results_path)
    print_per_query(data["per_query_results"])
    print_aggregate(data["aggregated"])


if __name__ == "__main__":
    main(sys.argv[1])
