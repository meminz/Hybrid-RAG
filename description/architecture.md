# Hybrid RAG System - Technical Architecture

## Overview

A Retrieval-Augmented Generation (RAG) system that combines semantic search (embeddings) with keyword search (BM25) to retrieve relevant document context and generate answers using a local LLM via Ollama. Supports multiple document splitting strategies (full documents, paragraphs, heading chunks, fixed-size chunks), version-aware retrieval with automatic version boosting, metadata-based score boosting, and an interactive chat interface with streaming output.

## RAG Workflow

<img src="architecture-diagram.png" width="600" alt="RAG Workflow">

## Requirements

### Prerequisites

- **Python 3.14+**
- **Ollama** running locally (default: http://localhost:11434)
- **qwen2.5** model loaded in Ollama (or configure via `OLLAMA_MODEL` env var)

### Dependencies

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. All required packages are defined in `pyproject.toml`:

- `numpy>=2.4.4`
- `ollama>=0.6.1`
- `rank-bm25>=0.2.2`
- `sentence-transformers>=5.3.0`

### Setup

Install all dependencies
```bash
uv sync
```

This creates a virtual environment and installs all dependencies automatically.

## Command Line Interface

### Usage

```python
python main.py <folder> [OPTIONS]
```

### Arguments
```
positional arguments:
  folder                Path to folder containing documents

options:
  -h, --help            show this help message and exit
  --max-docs MAX_DOCS   Maximum number of documents to load (default: unlimited)
  --paragraphs          Split documents into paragraphs for finer retrieval
  --headings            Split documents by markdown headings (## level)
  --heading-level N     Minimum heading level to split on (2=##, 3=###, default: 2)
  --chunk-size N        Split documents into fixed-size chunks (in tokens, default: no chunking)
  --overlap N           Number of overlapping tokens between chunks (default: 50)
  --top-k K             Number of documents to retrieve per query (default: 5)
  --metadata-boost      Enable metadata-based score boosting
  --version V           Boost specific version in results (e.g. 'v52', default: latest)
```

### Examples

```bash
# Basic usage (full documents)
python main.py path/to/your/wiki/

# Paragraph-level retrieval (recommended for large docs)
python main.py path/to/your/wiki/ --paragraphs --top-k 3

# Heading-based chunking (default)
python main.py path/to/your/wiki/ --headings --top-k 3

# Fixed-size chunking
python main.py path/to/your/wiki/ --chunk-size 500 --top-k 3

# Limit documents for faster startup
python main.py path/to/your/wiki/ --max-docs 50 --top-k 2

# Enable metadata boosting for filename matches
python main.py path/to/your/wiki/ --paragraphs --metadata-boost

# Boost older version explicitly
python main.py path/to/wiki-datasets/ --paragraphs --version v1
```

## File Structure

```
src/
├── main.py                 # Entry point, CLI, chat loop
├── RagClass.py             # Hybrid RAG implementation
├── RagEvaluator.py         # Evaluation system (3-method comparison)
├── ollama_client.py        # Ollama LLM integration
├── terminal_ui.py          # Loading animations, streaming display
├── _utils.py               # Document loading, chunking, metadata extraction
├── arg_parser.py           # CLI argument parsing
└── evaluation.py           # Evaluation runner
```

## Metadata System

Every document and paragraph automatically includes extracted metadata from the folder structure. This is mandatory - all documents always have metadata.

### Metadata Extraction (_utils.py)

`extract_metadata_from_path()` analyzes file paths to extract:
- **dataset:** Dataset folder name (e.g., 'my-project-v1')
- **project:** Project name (e.g., 'my-project', 'other-project') - always present
- **version:** Version if present (e.g., 'v1', 'v0.52.0', 'v1.2.3')
- **category:** Parent folder (e.g., 'docs', 'guides')
- **source:** Full file path
- **filename:** Base filename
- **rel_path:** Path relative to load folder

**Smart extraction:**
- If loading from 'wiki-datasets/my-project-v1/', extracts from 'my-project-v1'
- If loading from 'wiki-datasets/', extracts from subfolder (e.g., 'my-project-v1')

**Version patterns supported:**
- Simple: v1, v2, v10
- Semantic: v0.52.0, v1.2.3
- If no version found, 'version' field is omitted (not set to None)

### Metadata for Paragraphs

When using `--paragraphs`, each paragraph inherits all source metadata plus:
- **paragraph:** Paragraph index (0-based)
- **total_paragraphs:** Total paragraphs in source document

### Metadata for Heading Chunks

When using `--headings`, each heading chunk inherits all source metadata plus:
- **heading:** The heading text (without # symbols)
- **chunk:** Chunk index (0-based)
- **total_chunks:** Total heading chunks in source document

### Metadata for Fixed-Size Chunks

When using `--chunk-size`, each chunk inherits all source metadata plus:
- **chunk:** Chunk index (0-based)
- **total_chunks:** Total chunks in source document

### Example Metadata (paragraph)

```json
{
  "source": "wiki-datasets/my-project-v1/docs/guide.md",
  "filename": "guide.md",
  "rel_path": "docs/guide.md",
  "dataset": "my-project-v1",
  "project": "my-project",
  "version": "v1",
  "category": "docs",
  "paragraph": 5,
  "total_paragraphs": 42
}
```

### Example Metadata (no version)

```json
{
  "source": "wiki-datasets/other-project/getting-started.md",
  "filename": "getting-started.md",
  "rel_path": "getting-started.md",
  "dataset": "other-project",
  "project": "other-project",
  "category": "basics"
}
```

Note: 'version' field is omitted when not found.

### Display Format

Retrieved documents show metadata in a structured format:

```
[my-project-v1/guide.md#para5]
my-project | v1 | docs | [Guide] | para 5/42
RRF Score: 0.016667
```

For heading chunks:

```
[my-project-v2/window-management.md#heading2]
my-project | v2 | [Window Management] | chunk 2/8
RRF Score: 0.016667
```

### Version Boosting (Default Behavior)

When multiple versions are loaded, the latest version is automatically detected and boosted (1.5x score multiplier) to prioritize recent content.

**Auto-detection:** `_utils.get_latest_version()` parses version strings and compares them numerically (e.g., v2 > v1, v0.53.0 > v0.52.0).

**Query version detection:** `_utils.detect_version_in_query()` scans queries for version patterns (v1, v2, version 1, my-project 2) and applies stronger boosting (2.25x) to matching documents while penalizing others (0.7x).

To override and boost a specific version:

```bash
python main.py path/to/wiki-datasets/ --paragraphs --version v1
```

Without `--version` flag, latest version is always boosted by default.

**Metadata boosting** (`--metadata-boost` flag):
- Matches query tokens against document filenames
- Normalizes hyphens/underscores for better matching
- Applies 2.0x per token match, 2.5x for normalized full match
- Helps navigational queries like "window management" match window-management.md

## Evaluation System

Evaluate retrieval quality with version-specific queries:

```bash
python evaluation.py
```

- Queries loaded from `evaluation_queries.txt`
- Format: `query | expected_version | relevant_docs`
	- relevant_docs: semicolon-separated "doc_path:grade" pairs
	- grade 2 = primary answer, grade 1 = supplementary
- Compares 3 methods: Hybrid RRF, BM25-only, Embedding-only
- All methods operate on same heading-chunk indices
- Metrics: Precision@K, Recall@K, NDCG@K (graded), MRR
- Results saved to `evaluation_results.json`
- Report includes: method comparison table, best-per-method per metric, version-based breakdown, worst-performing queries analysis
