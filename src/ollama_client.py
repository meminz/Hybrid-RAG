"""
Ollama Integration for Text Generation

This module provides functions to interact with Ollama for generating
responses in the RAG pipeline with streaming support.

Ollama must be running locally (default: http://localhost:11434)
"""

import os
from typing import List, Dict, Tuple, Optional, Generator
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

# Ollama host (default: localhost:11434)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Default model to use
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5")

# Default generation parameters
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.7


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_grounded_prompt(
    query: str,
    retrieved_docs: List[Dict],
    system_instruction: str = None
) -> str:
    """
    Build a prompt that grounds the response in retrieved context.
    
    Args:
        query: User question
        retrieved_docs: List of retrieved document dictionaries with 'id' and 'content'
        system_instruction: Optional custom instruction for the model
        
    Returns:
        Formatted prompt string
    """
    if system_instruction is None:
        system_instruction = (
            "You are a helpful assistant that must answer questions using ONLY "
            "the provided context. If the context does not contain enough information "
            "to answer the question, say so clearly. Do not make up information."
        )
    
    # Format context with human-readable reference labels
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        doc_id = doc.get('id', f'Doc {i}')
        content = doc.get('content', doc.get('text', ''))
        score = doc.get('rrf_score') or doc.get('score')
        metadata = doc.get('metadata', {})

        # Build a concise, user-friendly label from metadata
        label_parts = []
        if 'filename' in metadata:
            label_parts.append(metadata['filename'])
        if 'heading' in metadata:
            label_parts.append(metadata['heading'])
        if 'version' in metadata:
            label_parts.append(f"({metadata['version']})")

        ref_label = " ".join(label_parts) if label_parts else doc_id

        if score is not None:
            context_parts.append(f"[Source {i + 1}: {ref_label} | score={score:.4f}]\n{content}")
        else:
            context_parts.append(f"[Source {i + 1}: {ref_label}]\n{content}")

    context_text = "\n\n".join(context_parts)

    prompt = (
        f"{system_instruction}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer concisely and cite your sources by their label "
        "(e.g., 'Window-Rules.md Syntax (v53)' or 'Source 1').\n"
    )
    
    return prompt


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def generate_answer_streaming(
    query: str,
    retrieved_docs: List[Dict],
    model: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    system_instruction: str = None,
    host: str = None
) -> Generator[str, None, Tuple[str, Dict]]:
    """
    Generate an answer with streaming output (token by token).
    
    This is a generator function that yields tokens as they arrive.
    Use it like:
        for token in generate_answer_streaming(...):
            print(token, end="", flush=True)
    
    Args:
        query: User question
        retrieved_docs: List of retrieved documents
        model: Model name to use (e.g., 'llama3.2', 'gemma2')
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_instruction: Optional custom system instruction
        host: Override for Ollama host URL
        
    Yields:
        Individual tokens/chunks of text as they are generated
        
    Returns:
        Tuple of (full_answer_text, metadata_dict)
    """
    model_name = model or DEFAULT_MODEL
    
    # Set host if provided
    if host:
        ollama_client = ollama.Client(host=host)
    else:
        ollama_client = ollama.Client()
    
    prompt = build_grounded_prompt(query, retrieved_docs, system_instruction)
    
    full_text = ""
    metadata = {}
    
    try:
        # Stream response
        stream = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "num_predict": max_tokens,
                "temperature": temperature
            },
            stream=True
        )
        
        for chunk in stream:
            if chunk:
                content = chunk["message"].get("content", "")
                if content:
                    full_text += content
                    yield content
                
                # Collect metadata from final chunk
                if chunk.get("done", False):
                    metadata = {
                        "model": chunk.get("model", model_name),
                        "total_duration": chunk.get("total_duration"),
                        "load_duration": chunk.get("load_duration"),
                        "prompt_eval_count": chunk.get("prompt_eval_count"),
                        "eval_count": chunk.get("eval_count"),
                        "eval_duration": chunk.get("eval_duration")
                    }
        
        return full_text, metadata
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nPlease ensure Ollama is running with the model pulled."
        yield error_msg
        return error_msg, {"error": str(e)}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_connection(host: str = None) -> bool:
    """
    Check if Ollama server is accessible.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        if host:
            client = ollama.Client(host=host)
        else:
            client = ollama.Client()

        client.list()
        return True
    except Exception:
        return False


def get_loaded_model_info(host: str = None) -> Optional[Dict]:
    """
    Get information about available models in Ollama.

    Returns:
        Dict with model info or None if not available
    """
    try:
        if host:
            client = ollama.Client(host=host)
        else:
            client = ollama.Client()

        models = client.list()
        if models and models.get("models"):
            return {
                "name": models["models"][0].get("name", "unknown"),
                "size": models["models"][0].get("size"),
                "modified_at": models["models"][0].get("modified_at")
            }
    except Exception:
        pass

    return None
