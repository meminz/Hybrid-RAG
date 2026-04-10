"""
Evaluation Script for Hybrid RAG System

This script evaluates the hybrid retrieval system against:
1. Embedding-only retrieval
2. BM25-only retrieval

All three methods operate on the exact same heading-chunk indices and parameters.
The evaluator delegates to the HybridRAG's internal _bm25_search and _embedding_search
so there is no index duplication or parameter drift.

Metrics:
- Precision@K
- Recall@K (against full set of relevant docs)
- NDCG@K (Normalized Discounted Cumulative Gain with graded relevance)
- MRR (Mean Reciprocal Rank)

Usage:
    python evaluate_hybrid_rag.py
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import _utils
from RagClass import HybridRAG
import json


# ============================================================
# EVALUATION METRICS
# ============================================================

def _doc_matches(doc_id: str, relevant_paths: Set[str]) -> bool:
    """Check if a document ID contains any of the relevant path substrings."""
    return any(path in doc_id for path in relevant_paths)


def _get_doc_grade(doc_id: str, relevant_docs: Dict[str, int]) -> int:
    """
    Get the relevance grade for a document.
    Returns the grade (2=primary, 1=supplementary, 0=not relevant).
    """
    for path, grade in relevant_docs.items():
        if path in doc_id:
            return grade
    return 0


def precision_at_k(retrieved_ids: List[str], relevant_paths: Set[str], k: int) -> float:
    """
    Calculate Precision@K: fraction of retrieved documents that are relevant.
    """
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if _doc_matches(doc_id, relevant_paths))
    return relevant_retrieved / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_paths: Set[str], k: int) -> float:
    """
    Calculate Recall@K: fraction of relevant documents that were retrieved.
    """
    total_relevant = len(relevant_paths)
    if total_relevant == 0:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if _doc_matches(doc_id, relevant_paths))
    return relevant_retrieved / total_relevant


def dcg_at_k(gains: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain @ K.
    DCG@K = sum(gain_i / log2(i + 2)) for i = 0 to K-1
    """
    gains = gains[:k]
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    """
    Calculate Normalized DCG @ K with graded relevance.
    Uses the actual grades (2 or 1) as gain values.
    """
    if not relevant_docs:
        return 0.0

    retrieved_gains = [_get_doc_grade(doc_id, relevant_docs) for doc_id in retrieved_ids[:k]]
    dcg = dcg_at_k(retrieved_gains, k)

    # Ideal DCG: sort all relevant docs by grade descending, take top k
    all_grades = sorted(relevant_docs.values(), reverse=True)
    ideal_gains = all_grades[:k]
    idcg = dcg_at_k(ideal_gains, k)

    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(retrieved_ids: List[str], relevant_paths: Set[str], k: int) -> float:
    """
    Calculate Reciprocal Rank @ K: 1/rank of first relevant document.
    """
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if _doc_matches(doc_id, relevant_paths):
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# EVALUATION CLASS
# ============================================================

class HybridRAGEvaluator:
    """
    Evaluator for comparing hybrid, embedding-only, and BM25-only retrieval.

    All three methods use the same underlying heading-chunk indices from the
    HybridRAG instance. The evaluator calls the RAG's internal _bm25_search
    and _embedding_search methods directly, so there is no index duplication
    or parameter mismatch.
    """

    def __init__(self, rag_system: HybridRAG, use_metadata_boost: bool = False):
        self.rag = rag_system
        self.documents = rag_system.documents
        self.use_metadata_boost = use_metadata_boost

    def _get_relevant_paths(self, query_info: Dict) -> Set[str]:
        """Extract the set of relevant doc paths from query info."""
        return set(query_info.get('relevant_docs', {}).keys())

    def _get_ranking(self, method: str, query: str, top_k: int,
                     detected_version: str = None) -> List[str]:
        """
        Get a ranked list of document IDs for a given method.

        All methods operate on the same heading chunks:
        - 'hybrid':    full hybrid RRF pipeline (via rag.retrieve)
        - 'bm25':      BM25-only (via rag._bm25_search)
        - 'embedding': embedding-only (via rag._embedding_search)
        """
        if method == 'hybrid':
            results = self.rag.retrieve(
                query, top_k=top_k, return_scores=False,
                detected_version=detected_version,
                use_metadata_boost=self.use_metadata_boost
            )
            return [doc['id'] for doc in results]

        elif method == 'bm25':
            ranked = self.rag._bm25_search(query)
            return [self.documents[idx]['id'] for idx, _ in ranked[:top_k]]

        elif method == 'embedding':
            ranked = self.rag._embedding_search(query)
            return [self.documents[idx]['id'] for idx, _ in ranked[:top_k]]

        else:
            raise ValueError(f"Unknown method: {method}")

    def evaluate_query(
        self,
        query_info: Dict,
        k_values: List[int] = [1, 2, 3]
    ) -> Dict:
        """
        Evaluate all three methods on a single query.
        """
        query = query_info['query']
        relevant_docs = query_info.get('relevant_docs', {})
        relevant_paths = self._get_relevant_paths(query_info)
        expected_version = query_info.get('expected_version', '')

        detected_version = _utils.detect_version_in_query(query, self.documents)
        max_k = max(k_values)

        # Get rankings from each method
        rankings = {}
        for method in ['hybrid', 'bm25', 'embedding']:
            rankings[method] = self._get_ranking(method, query, max_k, detected_version)

        results = {
            'query': query,
            'description': query_info.get('description', ''),
            'expected_version': expected_version,
            'relevant_docs': relevant_docs,
            'num_relevant': len(relevant_paths),
            'metrics': {}
        }

        for method_name, ranking in rankings.items():
            method_metrics = {}

            for k in k_values:
                method_metrics[f'precision@{k}'] = precision_at_k(ranking, relevant_paths, k)
                method_metrics[f'recall@{k}'] = recall_at_k(ranking, relevant_paths, k)
                method_metrics[f'ndcg@{k}'] = ndcg_at_k(ranking, relevant_docs, k)

            method_metrics['mrr'] = mrr_at_k(ranking, relevant_paths, max_k)

            # Track which specific heading chunks were retrieved (for breakdown)
            retrieved_relevant = []
            for rank_pos, doc_id in enumerate(ranking[:max_k], 1):
                grade = _get_doc_grade(doc_id, relevant_docs)
                if grade > 0:
                    for path in relevant_paths:
                        if path in doc_id:
                            retrieved_relevant.append({
                                'path': path,
                                'grade': grade,
                                'rank': rank_pos,
                                'doc_id': doc_id
                            })
                            break

            results['metrics'][method_name] = method_metrics
            results[f'{method_name}_ranking'] = ranking
            results[f'{method_name}_retrieved_relevant'] = retrieved_relevant

        return results

    def evaluate_all(
        self,
        queries: List[Dict],
        k_values: List[int] = [1, 2, 3]
    ) -> Dict:
        """Evaluate all queries and aggregate results."""
        all_results = []

        print("\n" + "=" * 70)
        print("EVALUATING RETRIEVAL METHODS")
        print("=" * 70)

        for i, query_info in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {query_info['description']}")
            print(f"    Query: {query_info['query']}")
            print(f"    Expected version: {query_info.get('expected_version', 'N/A')}")
            print(f"    Relevant docs: {len(query_info.get('relevant_docs', {}))}")

            result = self.evaluate_query(query_info, k_values)
            all_results.append(result)

            for method in ['hybrid', 'bm25', 'embedding']:
                p1 = result['metrics'][method].get('precision@1', 0)
                r1 = result['metrics'][method].get('recall@1', 0)
                n1 = result['metrics'][method].get('ndcg@1', 0)
                print(f"    {method.upper():10} P@1={p1:.2f} R@1={r1:.2f} NDCG@1={n1:.2f}")

        aggregated = self._aggregate_results(all_results, k_values)

        return {
            'per_query_results': all_results,
            'aggregated': aggregated
        }

    def _aggregate_results(self, results: List[Dict], k_values: List[int]) -> Dict:
        """Aggregate metrics across all queries."""
        methods = ['hybrid', 'bm25', 'embedding']

        aggregated = {
            'num_queries': len(results),
            'methods': {}
        }

        for method in methods:
            method_data = {'k_values': k_values}

            for k in k_values:
                method_data[f'avg_precision@{k}'] = np.mean([
                    r['metrics'][method][f'precision@{k}'] for r in results
                ])
                method_data[f'avg_recall@{k}'] = np.mean([
                    r['metrics'][method][f'recall@{k}'] for r in results
                ])
                method_data[f'avg_ndcg@{k}'] = np.mean([
                    r['metrics'][method][f'ndcg@{k}'] for r in results
                ])

            method_data['avg_mrr'] = np.mean([
                r['metrics'][method]['mrr'] for r in results
            ])

            aggregated['methods'][method] = method_data

        aggregated['comparison'] = self._compare_methods(results, methods)
        return aggregated

    def _compare_methods(self, results: List[Dict], methods: List[str]) -> Dict:
        """Compare methods to determine which performs best."""
        comparison = {}

        for metric in ['precision@1', 'precision@2', 'precision@3',
                       'recall@1', 'recall@2', 'recall@3',
                       'ndcg@1', 'ndcg@2', 'ndcg@3', 'mrr']:

            if metric == 'mrr':
                scores = {
                    method: np.mean([r['metrics'][method]['mrr'] for r in results])
                    for method in methods
                }
            else:
                k = int(metric.split('@')[1])
                key = f'{metric.split("@")[0]}@{k}'
                scores = {
                    method: np.mean([r['metrics'][method][key] for r in results])
                    for method in methods
                }

            best_method = max(scores, key=scores.get)
            comparison[metric] = {
                'scores': scores,
                'best': best_method,
                'best_score': scores[best_method]
            }

        return comparison

    def print_report(self, evaluation_results: Dict) -> None:
        """Print a formatted evaluation report."""
        aggregated = evaluation_results['aggregated']
        all_results = evaluation_results['per_query_results']

        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)

        print(f"\nTotal queries evaluated: {aggregated['num_queries']}")

        # Method comparison table
        print("\n" + "-" * 70)
        print("METHOD COMPARISON (Averages)")
        print("-" * 70)

        methods = ['hybrid', 'bm25', 'embedding']
        k_values = aggregated['methods']['hybrid']['k_values']

        header = f"{'Method':<12}"
        for k in k_values:
            header += f"  P@{k:<4}"
        for k in k_values:
            header += f"  R@{k:<4}"
        for k in k_values:
            header += f"  N@{k:<4}"
        header += f"  MRR"
        print(header)
        print("-" * 70)

        for method in methods:
            data = aggregated['methods'][method]
            row = f"{method.upper():<12}"
            for k in k_values:
                row += f"  {data[f'avg_precision@{k}']:.3f} "
            for k in k_values:
                row += f"  {data[f'avg_recall@{k}']:.3f} "
            for k in k_values:
                row += f"  {data[f'avg_ndcg@{k}']:.3f} "
            row += f"  {data['avg_mrr']:.3f}"
            print(row)

        # Best method per metric
        print("\n" + "-" * 70)
        print("BEST METHOD PER METRIC")
        print("-" * 70)

        for metric, data in aggregated['comparison'].items():
            best = data['best'].upper()
            score = data['best_score']
            all_scores = ", ".join([f"{m.upper()}={s:.3f}" for m, s in data['scores'].items()])
            print(f"{metric:<12}: {best} ({score:.3f}) | {all_scores}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        win_counts = defaultdict(int)
        for metric, data in aggregated['comparison'].items():
            win_counts[data['best']] += 1

        print("\nMethod wins across all metrics:")
        for method in methods:
            print(f"  {method.upper()}: {win_counts[method]} / {len(aggregated['comparison'])} metrics")

        best_overall = max(win_counts, key=win_counts.get)
        print(f"\nBest overall method: {best_overall.upper()}")

        # Version-based breakdown
        print("\n" + "-" * 70)
        print("VERSION RETRIEVAL BREAKDOWN")
        print("-" * 70)

        buckets = {'v52': [], 'v53': []}
        for result in all_results:
            expected = result.get('expected_version', '')
            if expected in buckets:
                m = result['metrics']['hybrid']
                buckets[expected].append({
                    'p1': m.get('precision@1', 0),
                    'r1': m.get('recall@1', 0),
                    'n1': m.get('ndcg@1', 0),
                    'mrr': m.get('mrr', 0),
                })

        for label, key in [("V52 (explicit version)", 'v52'), ("V53 (latest, boosted)", 'v53')]:
            bucket = buckets[key]
            if bucket:
                n = len(bucket)
                print(f"\n{label}:")
                print(f"  Count: {n}")
                print(f"  Avg Precision@1: {sum(b['p1'] for b in bucket)/n:.3f}")
                print(f"  Avg Recall@1:    {sum(b['r1'] for b in bucket)/n:.3f}")
                print(f"  Avg NDCG@1:      {sum(b['n1'] for b in bucket)/n:.3f}")
                print(f"  Avg MRR:         {sum(b['mrr'] for b in bucket)/n:.3f}")

        # Query-by-query breakdown (worst performers)
        print("\n" + "-" * 70)
        print("WORST PERFORMING QUERIES (by hybrid NDCG@3)")
        print("-" * 70)

        sorted_by_ndcg = sorted(all_results, key=lambda r: r['metrics']['hybrid'].get('ndcg@3', 0))
        worst_n = min(10, len(sorted_by_ndcg))

        for result in sorted_by_ndcg[:worst_n]:
            ndcg3 = result['metrics']['hybrid'].get('ndcg@3', 0)
            p1 = result['metrics']['hybrid'].get('precision@1', 0)
            print(f"\n  NDCG@3={ndcg3:.3f} P@1={p1:.3f} | {result['query']}")
            print(f"    Expected: {result.get('description', 'N/A')}")

            for method in ['hybrid', 'bm25', 'embedding']:
                retrieved_relevant = result.get(f'{method}_retrieved_relevant', [])
                ranking = result.get(f'{method}_ranking', [])
                if retrieved_relevant:
                    doc_strs = [f"#{r['rank']} {r['path']}(g={r['grade']})" for r in retrieved_relevant]
                    print(f"    {method.upper():10} found: {', '.join(doc_strs)}")
                else:
                    top3_ids = ranking[:3]
                    top3_short = [tid.split('/')[-1] for tid in top3_ids]
                    print(f"    {method.upper():10} found: NONE (top: {', '.join(top3_short)})")

        print("\n" + "=" * 70)
