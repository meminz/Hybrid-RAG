import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG System - Interactive Chat"
    )

    # Required arguments
    parser.add_argument(
        "folder",
        help="Path to folder containing documents"
    )

    # Document loading options
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to load (default: unlimited)"
    )
    parser.add_argument(
        "--paragraphs",
        action="store_true",
        help="Split documents into paragraphs for finer retrieval"
    )
    parser.add_argument(
        "--headings",
        action="store_true",
        help="Split documents by markdown headings (## level)"
    )
    parser.add_argument(
        "--heading-level",
        type=int,
        default=2,
        help="Minimum heading level to split on (2=##, 3=###). Default: 2"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Split documents into fixed-size chunks (in tokens, default: no chunking)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping tokens between chunks (default: 50)"
    )

    # Retrieval options
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query (default: 5)"
    )
    parser.add_argument(
        "--metadata-boost",
        action="store_true",
        help="Enable metadata-based score boosting (matches query terms to filenames)"
    )

    # Version options
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Boost specific version in results (e.g., 'v52')"
    )

    return parser


def parse_args():
    """Parse command-line arguments."""
    parser = create_parser()
    return parser.parse_args()
