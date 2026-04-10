"""
Hybrid RAG System: Combining Embeddings (Semantic Search) + BM25 (Keyword Search)

This module implements a Retrieval-Augmented Generation system that uses:
1. Sentence Transformers for semantic embeddings
2. BM25 for keyword-based retrieval
3. Reciprocal Rank Fusion (RRF) to combine both approaches

Author: Hybrid RAG Implementation
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import pickle
import hashlib
import os
import _utils


# ============================================================
# CONFIGURATION
# ============================================================

# Embedding model for semantic search
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BM25 parameters
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# Retrieval parameters
TOP_K = 5  # Number of documents to retrieve
RRF_K = 60  # RRF ranking constant

# Weights for hybrid fusion (0.5 = equal weight)
EMBEDDING_WEIGHT = 0.7
BM25_WEIGHT = 0.3


# ============================================================
# HYBRID RAG CLASS
# ============================================================

class HybridRAG:
    """
    Hybrid Retrieval-Augmented Generation system combining:
    - Semantic search (Sentence Transformers embeddings)
    - Keyword search (BM25)
    - Reciprocal Rank Fusion for combining results
    """
    
    def __init__(
        self,
        documents: list[dict],
        embedding_model_name: str = EMBED_MODEL_NAME,
        bm25_k1: float = BM25_K1,
        bm25_b: float = BM25_B,
        top_k: int = TOP_K,
        rrf_k: int = RRF_K,
        embedding_weight: float = EMBEDDING_WEIGHT,
        bm25_weight: float = BM25_WEIGHT
    ):
        """
        Initialize the Hybrid RAG system.
        
        Args:
            documents: list of document dictionaries with 'id' and 'content'
            embedding_model_name: Name of the sentence transformer model
            bm25_k1: BM25 term frequency saturation parameter
            bm25_b: BM25 length normalization parameter
            top_k: Number of documents to retrieve
            rrf_k: RRF ranking constant
            embedding_weight: Weight for embedding scores in hybrid fusion
            bm25_weight: Weight for BM25 scores in hybrid fusion
        """
        if not documents:
            raise ValueError("No documents provided to HybridRAG. Check that your document folder contains .md or .txt files.")
        
        self.documents = documents
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight


        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)

        # Create or load cached embeddings
        self.doc_texts = [doc['content'] for doc in documents]
        cache_file = self._get_cache_file_path(documents, embedding_model_name)

        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.doc_embeddings = pickle.load(f)
        else:
            print("Computing document embeddings...")
            self.doc_embeddings = self.embedder.encode(
                self.doc_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            if cache_file:
                self._save_embeddings_to_cache(cache_file)

        # Create BM25 index
        print("Building BM25 index...")
        self.tokenized_docs = [_utils.tokenize(text) for text in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=bm25_k1, b=bm25_b)

        print("Hybrid RAG system initialized successfully!")


    def _get_cache_file_path(self, documents: list[dict], model_name: str) -> str:
        """
        Generate a cache file path based on document content hash and model name.

        Returns:
            Path to cache file, or None if cache directory doesn't exist
        """
        cache_dir = ".cache"

        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
            except Exception:
                return None

        doc_ids = "".join(doc['id'] for doc in documents)
        content_hash = hashlib.md5(doc_ids.encode()).hexdigest()
        model_safe = model_name.replace("/", "_")

        return os.path.join(cache_dir, f"embeddings_{model_safe}_{content_hash}.pkl")

    def _save_embeddings_to_cache(self, cache_file: str):
        """Save embeddings to cache file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.doc_embeddings, f)
            print(f"Embeddings cached to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not cache embeddings: {e}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _embedding_search(self, query: str) -> list[tuple[int, float]]:
        """
        Perform semantic search using embeddings.
        
        Returns:
            list of (doc_index, score) tuples sorted by similarity
        """
        query_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        similarities = [
            self._cosine_similarity(query_emb, doc_emb)
            for doc_emb in self.doc_embeddings
        ]
        
        ranked = sorted(
            enumerate(similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked
    
    def _bm25_search(self, query: str) -> list[tuple[int, float]]:
        """
        Perform keyword search using BM25.

        Returns:
            list of (doc_index, score) tuples sorted by BM25 score
        """
        query_tokens = _utils.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)
        
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked
    
    def _metadata_boost(self, query: str, doc_idx: int) -> float:
        """
        Apply a boost factor when query terms appear in document filename.

        Only filename matches are boosted -- heading and category matches
        are too noisy (e.g., the word "syntax" appearing in any heading).
        Filename matches are a much stronger signal for navigational queries.

        Handles hyphenated filenames by normalizing both sides for matching.

        Returns:
            Boost multiplier (1.0 = no boost)
        """
        doc = self.documents[doc_idx]
        metadata = doc.get('metadata', {})
        filename = metadata.get('filename', '').lower()

        query_tokens = _utils.tokenize(query)
        query_lower = query.lower()

        # Remove extension and hyphens for matching
        filename_no_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
        filename_normalized = filename_no_ext.replace('-', '').replace('_', '')

        boost = 1.0

        for token in query_tokens:
            # Normalize token too (remove hyphens/underscores)
            token_norm = token.replace('-', '').replace('_', '')
            # Check both exact and normalized
            if token in filename_no_ext or token_norm in filename_normalized:
                boost *= 2.0

        # Also check normalized full query against normalized filename
        query_normalized = query_lower.replace('-', '').replace('_', '').replace(' ', '')
        if query_normalized in filename_normalized or filename_normalized in query_normalized:
            boost *= 2.5

        return boost

    def _reciprocal_rank_fusion(
        self,
        embedding_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k: int = None
    ) -> list[tuple[int, float]]:
        """
        Combine ranking results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: sum(1 / (k + rank)) for each ranking
        
        Args:
            embedding_results: Ranked results from embedding search
            bm25_results: Ranked results from BM25 search
            k: RRF constant (higher = more emphasis on top ranks)
            
        Returns:
            Combined and re-ranked results
        """
        if k is None:
            k = self.rrf_k
        
        rrf_scores = {}
        
        # Create rank mappings (doc_idx -> rank)
        embedding_ranks = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(embedding_results)}
        bm25_ranks = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(bm25_results)}
        
        # Get all document indices
        all_doc_indices = set(embedding_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF score for each document
        for doc_idx in all_doc_indices:
            emb_rank = embedding_ranks.get(doc_idx, float('inf'))
            bm25_rank = bm25_ranks.get(doc_idx, float('inf'))
            
            rrf_score = 0.0
            
            # Add embedding contribution
            if emb_rank != float('inf'):
                rrf_score += self.embedding_weight / (k + emb_rank)
            
            # Add BM25 contribution
            if bm25_rank != float('inf'):
                rrf_score += self.bm25_weight / (k + bm25_rank)
            
            rrf_scores[doc_idx] = rrf_score
        
        # Sort by RRF score
        combined = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return combined
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_metadata: Optional[dict] = None,
        boost_version: Optional[str] = None,
        boost_factor: float = 1.5,
        detected_version: Optional[str] = None,
        use_metadata_boost: bool = False
    ) -> list[dict]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: User query string
            top_k: Number of documents to return (overrides default)
            return_scores: Whether to include scores in results
            filter_metadata: Optional dict to filter by metadata fields
                           E.g., {'version': 'v52'} to only search v52 docs
            boost_version: Optional version to boost (e.g., 'v53')
            boost_factor: Score multiplier for boosted version (default: 1.5)
            detected_version: Version detected in query (used for intelligent boosting)
            use_metadata_boost: Whether to apply metadata-based score boosting

        Returns:
            list of retrieved documents with metadata and scores
        """
        if top_k is None:
            top_k = self.top_k

        # Get rankings from both methods
        embedding_results = self._embedding_search(query)
        bm25_results = self._bm25_search(query)

        # Combine using RRF
        combined_results = self._reciprocal_rank_fusion(embedding_results, bm25_results)

        # Apply metadata boost if enabled (helps match query terms to filenames)
        if use_metadata_boost:
            metadata_boosted = []
            for doc_idx, score in combined_results:
                boost = self._metadata_boost(query, doc_idx)
                metadata_boosted.append((doc_idx, score * boost))
            combined_results = sorted(metadata_boosted, key=lambda x: x[1], reverse=True)

        # Apply metadata filter if provided
        if filter_metadata:
            filtered_results = []
            for doc_idx, score in combined_results:
                doc = self.documents[doc_idx]
                match = all(
                    doc.get('metadata', {}).get(key) == value
                    for key, value in filter_metadata.items()
                )
                if match:
                    filtered_results.append((doc_idx, score))
            combined_results = filtered_results

        # Apply version-aware scoring
        if detected_version:
            # User explicitly mentioned a version - strongly prefer it
            version_results = []
            for doc_idx, score in combined_results:
                doc = self.documents[doc_idx]
                doc_version = doc.get('metadata', {}).get('version')
                if doc_version == detected_version:
                    score *= boost_factor * 1.5  # Strong boost for detected version
                elif doc_version:
                    score *= 0.7  # Penalize non-matching versions
                version_results.append((doc_idx, score))
            combined_results = sorted(version_results, key=lambda x: x[1], reverse=True)
        elif boost_version:
            # Apply standard version boost
            boosted_results = []
            for doc_idx, score in combined_results:
                doc = self.documents[doc_idx]
                doc_version = doc.get('metadata', {}).get('version')
                if doc_version == boost_version:
                    score *= boost_factor  # Boost the score
                boosted_results.append((doc_idx, score))
            # Re-sort after boosting
            combined_results = sorted(boosted_results, key=lambda x: x[1], reverse=True)

        # Select top-k
        top_results = combined_results[:top_k]

        # Format results
        results = []
        for doc_idx, rrf_score in top_results:
            doc = self.documents[doc_idx].copy()

            if return_scores:
                # Get individual scores for transparency
                emb_score = next((s for i, s in embedding_results if i == doc_idx), 0.0)
                bm25_score = next((s for i, s in bm25_results if i == doc_idx), 0.0)

                doc['retrieval_scores'] = {
                    'rrf_score': rrf_score,
                    'embedding_score': emb_score,
                    'bm25_score': bm25_score,
                    'embedding_rank': next((r + 1 for r, (i, _) in enumerate(embedding_results) if i == doc_idx), None),
                    'bm25_rank': next((r + 1 for r, (i, _) in enumerate(bm25_results) if i == doc_idx), None)
                }

            results.append(doc)

        return results
