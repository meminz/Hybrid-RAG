# import sys
import _utils
import RagClass
import ollama_client
import terminal_ui
import arg_parser

if __name__ == "__main__":
    args = arg_parser.parse_args()

    print("=" * 60)
    print("HYBRID RAG SYSTEM - INTERACTIVE CHAT")
    print("=" * 60)

    # Load documents
    if args.headings:
        documents = _utils.load_heading_chunked_documents(
            args.folder,
            min_heading_level=args.heading_level,
            max_docs=args.max_docs
        )
        print(f"\nLoaded {len(documents)} heading chunks from {args.folder}")

    elif args.paragraphs:
        documents = _utils.load_paragraph_chunked_documents(args.folder, max_docs=args.max_docs)
        print(f"\nLoaded {len(documents)} paragraphs from {args.folder}")

    elif args.chunk_size:
        documents = _utils.load_chunked_documents(
            args.folder,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_docs=args.max_docs
        )
        print(f"\nLoaded {len(documents)} chunks from {args.folder}")
    else:
        documents = _utils.load_documents_from_folder(args.folder, max_docs=args.max_docs)
        print(f"\nLoaded {len(documents)} documents from {args.folder}")

    # Determine version boosting (default to latest version)
    boost_version = None
    if args.version:
        boost_version = args.version
        print(f"Boosting version: {boost_version}")
    else:
        # Auto-detect and boost latest version by default
        boost_version = _utils.get_latest_version(documents)
        if boost_version:
            print(f"Latest version detected: {boost_version} (boosting by default)")

    # Initialize hybrid RAG
    rag = RagClass.HybridRAG(documents=documents)

    # Check Ollama connection
    print("\nChecking Ollama connection...")
    if ollama_client.check_connection():
        model_info = ollama_client.get_loaded_model_info()
        if model_info:
            print(f"Connected to Ollama (model: {model_info['name']})")
        else:
            print("Connected to Ollama (no models listed)")
    else:
        print("Warning: Could not connect to Ollama. Generation will fail.")
        print("Start Ollama and pull a model, e.g.: ollama pull llama3.2")

    print("\n" + "=" * 60)
    print("Ask a question about the documents (type 'exit' to quit)")
    print("=" * 60)

    while True:
        # Get user input
        query = input("\nYour question: ").strip()

        # Check for exit command
        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        # Skip empty queries
        if not query:
            continue

        # Detect version in query
        detected_version = _utils.detect_version_in_query(query, documents)
        if detected_version:
            print(f"Detected version in query: {detected_version}")

        # Retrieve documents first
        retrieved = rag.retrieve(
            query,
            top_k=args.top_k,
            return_scores=True,
            boost_version=boost_version,
            detected_version=detected_version,
            use_metadata_boost=args.metadata_boost
        )

        print("\n" + "-" * 60)
        print("ANSWER:")
        print("-" * 60)

        # Stream the answer token by token with thinking indicator
        print()  # Ensure we're on a new line
        for token in terminal_ui.stream_with_indicator(
            ollama_client.generate_answer_streaming(query=query, retrieved_docs=retrieved),
            "Thinking"
        ):
            print(token, end="", flush=True)

        print()  # Newline after streaming

        print("\n" + "-" * 60)
        print("RETRIEVED DOCUMENTS:")
        print("-" * 60)
        for j, doc in enumerate(retrieved, 1):
            rrf_score = doc.get('retrieval_scores', {}).get('rrf_score', 0)
            metadata = doc.get('metadata', {})
            
            # Build metadata display
            meta_parts = []
            if 'project' in metadata:
                meta_parts.append(metadata['project'])
            if 'version' in metadata:
                meta_parts.append(metadata['version'])
            if 'category' in metadata:
                meta_parts.append(metadata['category'])
            if 'heading' in metadata:
                meta_parts.append(f"[{metadata['heading']}]")
            if 'paragraph' in metadata:
                meta_parts.append(f"para {metadata['paragraph']}/{metadata['total_paragraphs']}")
            if 'chunk' in metadata and 'heading' not in metadata:
                meta_parts.append(f"chunk {metadata['chunk']}/{metadata['total_chunks']}")
            
            meta_str = " | ".join(meta_parts) if meta_parts else "unknown"
            
            print(f"  {j}. [{doc['id']}]")
            print(f"     {meta_str}")
            print(f"     RRF Score: {rrf_score:.6f}\n")

        print("-" * 60)

    print("\nSession ended.")
