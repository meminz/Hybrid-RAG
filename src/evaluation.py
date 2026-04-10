from RagClass import HybridRAG
from RagEvaluator import HybridRAGEvaluator
import _utils
import os
import json
import numpy as np
from typing import List, Dict


def create_evaluation_queries(queries_file: str = None) -> List[Dict]:
    """
    Load and parse evaluation queries from a text file.

    The file format is:
    - Lines starting with # are comments
    - Empty lines are skipped
    - Query lines: query | version | relevant_docs

    relevant_docs is a semicolon-separated list of "doc_path:grade" pairs, e.g.:
        Window-Rules.md:2;Binds.md:1

    grade 2 = primary answer, grade 1 = supplementary

    The evaluator matches doc_path against document IDs (which contain the path).

    Args:
        queries_file: Path to the queries file

    Returns:
        List of query dictionaries with 'query', 'expected_version', 'relevant_docs'
        where relevant_docs is a dict {doc_path: grade}
    """
    if queries_file is None:
        queries_file = os.path.join(os.path.dirname(__file__), "evaluation_queries.txt")

    queries = []

    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                query = parts[0]
                expected_version = parts[1]
                relevant_docs_str = parts[2]

                # Parse relevant_docs: "path1:2;path2:1" -> {"path1": 2, "path2": 1}
                relevant_docs = {}
                for entry in relevant_docs_str.split(';'):
                    entry = entry.strip()
                    if ':' in entry:
                        doc_path, grade = entry.rsplit(':', 1)
                        relevant_docs[doc_path.strip()] = int(grade.strip())

                # Build a human-readable description
                doc_descs = []
                for path, grade in relevant_docs.items():
                    label = "primary" if grade == 2 else "supplementary"
                    doc_descs.append(f"{path} ({label})")

                queries.append({
                    'query': query,
                    'expected_version': expected_version,
                    'relevant_docs': relevant_docs,
                    'description': f"Expect {', '.join(doc_descs)} for {expected_version}"
                })

    return queries


if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID RAG SYSTEM EVALUATION")
    print("=" * 70)

    # Load documents
    RAGDOC_FOLDER = os.path.join(os.path.dirname(__file__), "..", "wiki-datasets/")
    RAGDOC_FOLDER = os.path.abspath(RAGDOC_FOLDER)

    if not os.path.exists(RAGDOC_FOLDER):
        print(f"Error: Document folder not found: {RAGDOC_FOLDER}")
        exit(1)

    documents = _utils.load_heading_chunked_documents(RAGDOC_FOLDER, min_heading_level=2)
    print(f"\nLoaded {len(documents)} heading chunks")

    # Initialize Hybrid RAG
    rag = HybridRAG(documents=documents)

    # Create evaluator
    evaluator = HybridRAGEvaluator(rag, use_metadata_boost=True)

    # Get evaluation queries
    eval_queries = create_evaluation_queries()
    
    v52_queries = [q for q in eval_queries if q.get('expected_version') == 'v52']
    v53_queries = [q for q in eval_queries if q.get('expected_version') == 'v53']
    print(f"\nV52 queries: {len(v52_queries)}, V53 queries: {len(v53_queries)}")

    # Run evaluation
    results = evaluator.evaluate_all(eval_queries, k_values=[1, 2, 3])

    # Print report
    evaluator.print_report(results)

    # Save results
    output_file = "evaluation_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    serializable_results = convert_numpy(results)

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
