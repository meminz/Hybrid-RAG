import os
import re
from typing import Optional, List


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Converts text to lowercase and extracts words.
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def normalize_version(version: str) -> str:
    """
    Normalize version string for comparison.
    
    Examples:
        '52' -> 'v52'
        'v52' -> 'v52'
        'v0.52.0' -> 'v0.52.0'
        '0.52.0' -> 'v0.52.0'
        'version 52' -> 'v52'
    """
    if not version:
        return ''
    
    # Remove common prefixes
    version = version.strip().lower()
    version = re.sub(r'^version[-_ ]*', 'v', version)
    version = re.sub(r'^ver[-_ ]*', 'v', version)
    version = re.sub(r'^v[-_ ]*', 'v', version)
    
    # Add 'v' prefix if missing
    if not version.startswith('v'):
        version = 'v' + version
    
    return version


def extract_metadata_from_path(filepath: str, folder_path: str) -> dict:
    """
    Extract metadata from file path and folder structure.

    Args:
        filepath: Full path to the file
        folder_path: Root folder path documents are loaded from

    Returns:
        Dictionary with extracted metadata fields
    """
    # Get relative path from the root folder
    rel_path = os.path.relpath(filepath, folder_path)
    path_parts = rel_path.split(os.sep)

    # Get the root folder name
    root_folder = os.path.basename(os.path.abspath(folder_path))

    # Determine the dataset folder to extract metadata from
    # If root folder looks like a dataset (contains -wiki- or version pattern), use it
    # Otherwise, use the first subfolder in the path
    dataset_folder = root_folder
    if not re.search(r'[-_]v\d+', root_folder) and '-wiki' not in root_folder:
        # Root doesn't look like a dataset folder, use first subfolder
        if len(path_parts) >= 1:
            dataset_folder = path_parts[0]

    # Extract metadata from path structure
    metadata = {
        'source': filepath,
        'filename': os.path.basename(filepath),
        'rel_path': rel_path,
        'dataset': dataset_folder,
    }

    # Extract category (parent folder of the file) if not in root
    if len(path_parts) > 1:
        metadata['category'] = path_parts[-2]  # Folder containing the file

    # Extract project and version from dataset folder name
    # Patterns: 'project-wiki-v0.52.0', 'project-wiki-v52', 'project-wiki', 'project'
    # Examples: 'hyprland-wiki-v0.52.0' -> project='hyprland', version='v0.52.0'
    #           'hyprland-wiki-v52' -> project='hyprland', version='v52'
    #           'neovim-wiki' -> project='neovim' (no version)
    #           'mydocs' -> project='mydocs' (no version)
    parts = dataset_folder.split('-')
    if len(parts) >= 1:
        # Project is always the first part
        metadata['project'] = parts[0]

    # Check for version pattern (e.g., 'v52', 'v0.52.0', 'v1.2.3', etc.)
    for part in parts:
        if re.match(r'^v\d+(\.\d+)*$', part):
            metadata['version'] = part
            break

    return metadata


def strip_frontmatter(content: str) -> str:
    """
    Remove YAML frontmatter from markdown content.

    Frontmatter is the block between --- delimiters at the start of a file.
    """
    content = content.strip()
    if content.startswith('---'):
        # Find the closing ---
        end_idx = content.find('---', 3)
        if end_idx != -1:
            content = content[end_idx + 3:].strip()
    return content


def load_documents_from_folder(folder_path: str, max_docs: int = None):
    """
    Load text documents from a folder recursively.

    Args:
        folder_path: Path to folder containing .txt or .md files
        max_docs: Maximum number of documents to load (None for unlimited)

    Returns:
        List of dictionaries with 'id', 'content', 'metadata'
    """
    documents = []

    for root, _, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.endswith('.txt') or filename.endswith('.md'):
                filepath = os.path.join(root, filename)

                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                content = strip_frontmatter(content)

                metadata = extract_metadata_from_path(filepath, folder_path)
                
                documents.append({
                    'id': filename,
                    'content': content,
                    'metadata': metadata
                })

                if max_docs is not None and len(documents) >= max_docs:
                    return documents

    return documents


def split_into_paragraphs(content: str) -> list:
    """
    Split a document into paragraphs.

    Args:
        content: Document text to split

    Returns:
        List of paragraphs (non-empty text blocks separated by blank lines)
    """
    paragraphs = []
    for block in content.split('\n\n'):
        block = block.strip()
        if block:
            paragraphs.append(block)
    return paragraphs


def _detect_min_heading_level(content: str) -> int:
    """
    Detect the minimum (highest priority) heading level in a markdown document.
    
    Returns:
        The lowest heading level found (e.g., 2 for ##, 3 for ###), or None if no headings found
    """
    min_level = None
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            hash_count = 0
            for char in stripped:
                if char == '#':
                    hash_count += 1
                else:
                    break
            if hash_count > 0:
                if min_level is None or hash_count < min_level:
                    min_level = hash_count
    return min_level


def split_into_heading_chunks(content: str, min_heading_level: int = 2) -> list:
    """
    Split a markdown document into chunks based on heading structure.

    Each chunk starts with a heading at the specified minimum level (or higher)
    and includes all content until the next heading at that level or higher.
    
    If no headings at the specified level are found, automatically falls back
    to deeper heading levels present in the document.

    Args:
        content: Markdown document text
        min_heading_level: Minimum heading level to split on (2 = '##', 3 = '###')

    Returns:
        List of text chunks, each starting with a heading
    """
    lines = content.split('\n')
    chunks = []
    current_chunk_lines = []
    current_heading = None

    # First, try with the requested min_heading_level
    # If that produces no splits, auto-detect and use deeper levels
    effective_level = min_heading_level
    
    # Check if we have any headings at or above the minimum level
    has_headings_at_level = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            hash_count = 0
            for char in stripped:
                if char == '#':
                    hash_count += 1
                else:
                    break
            if 0 < hash_count <= min_heading_level:
                has_headings_at_level = True
                break
    
    # If no headings found at the requested level, auto-detect the deepest level
    if not has_headings_at_level:
        detected_level = _detect_min_heading_level(content)
        if detected_level is not None:
            effective_level = detected_level

    heading_prefix = '#' * effective_level

    for line in lines:
        stripped = line.strip()

        # Check if this line is a heading at or above the effective level
        is_split_heading = False
        if stripped.startswith('#'):
            # Count the number of # symbols
            hash_count = 0
            for char in stripped:
                if char == '#':
                    hash_count += 1
                else:
                    break

            # Split if heading is at or above the effective level (lower number = higher level)
            if hash_count <= effective_level and hash_count > 0:
                is_split_heading = True

        if is_split_heading:
            # Save the previous chunk if it exists
            if current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append(chunk_text)

            # Start a new chunk with this heading
            current_chunk_lines = [line]
            current_heading = stripped
        else:
            current_chunk_lines.append(line)

    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def load_heading_chunked_documents(folder_path: str, min_heading_level: int = 2, max_docs: int = None) -> list:
    """
    Load documents and split them into chunks based on markdown headings.

    Args:
        folder_path: Path to folder containing .txt or .md files
        min_heading_level: Minimum heading level to split on (2 = '##', 3 = '###')
        max_docs: Maximum number of source documents to load (None for unlimited)

    Returns:
        List of dictionaries with 'id', 'content', 'metadata' (one entry per heading chunk)
    """
    documents = load_documents_from_folder(folder_path, max_docs)
    heading_chunks = []

    for doc in documents:
        chunks = split_into_heading_chunks(doc['content'], min_heading_level)
        for i, chunk in enumerate(chunks):
            # Copy all metadata from source document
            chunk_metadata = doc['metadata'].copy()
            chunk_metadata['chunk'] = i
            chunk_metadata['total_chunks'] = len(chunks)

            # Extract heading from chunk (first line if it starts with #)
            first_line = chunk.split('\n')[0].strip()
            if first_line.startswith('#'):
                chunk_metadata['heading'] = first_line.lstrip('#').strip()

            # Include dataset in ID to avoid collisions across versions
            dataset = doc['metadata'].get('dataset', '')
            heading_chunks.append({
                'id': f"{dataset}/{doc['id']}#heading{i}",
                'content': chunk,
                'metadata': chunk_metadata
            })

    return heading_chunks


def load_paragraph_chunked_documents(folder_path: str, max_docs: int = None) -> list:
    """
    Load documents and split them into paragraphs.
    Each paragraph becomes an independent retrieval unit.

    Args:
        folder_path: Path to folder containing .txt or .md files
        max_docs: Maximum number of source documents to load (None for unlimited)

    Returns:
        List of dictionaries with 'id', 'content', 'metadata' (one entry per paragraph)
    """
    documents = load_documents_from_folder(folder_path, max_docs)
    paragraph_docs = []

    for doc in documents:
        paragraphs = split_into_paragraphs(doc['content'])
        for i, para in enumerate(paragraphs):
            # Copy all metadata from source document
            para_metadata = doc['metadata'].copy()
            para_metadata['paragraph'] = i
            para_metadata['total_paragraphs'] = len(paragraphs)
            
            # Include dataset in ID to avoid collisions across versions
            dataset = doc['metadata'].get('dataset', '')
            paragraph_docs.append({
                'id': f"{dataset}/{doc['id']}#para{i}",
                'content': para,
                'metadata': para_metadata
            })

    return paragraph_docs


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """
    Tokenize text and return each token with its start/end character offsets.

    Args:
        text: Input text

    Returns:
        List of (token, char_start, char_end) tuples.
        Offits are relative to the original text and preserve punctuation positions.
    """
    tokens = []
    text_lower = text.lower()
    for match in re.finditer(r'\b\w+\b', text_lower):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def chunk_document(content: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split a document into overlapping chunks based on token count.

    Uses the same regex tokenizer as BM25 so chunk boundaries align
    with actual token boundaries.  Punctuation and whitespace are
    preserved in the output by slicing the original text via character
    offsets.

    Args:
        content: Document text to split
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    token_entries = tokenize_with_offsets(content)

    if len(token_entries) <= chunk_size:
        return [content]

    chunks = []
    start = 0  # index into token_entries

    while start < len(token_entries):
        end = min(start + chunk_size, len(token_entries))

        # Try to break at a sentence boundary if we're not at the end
        if end < len(token_entries):
            # Look for the last period/newline within the token window
            # by scanning tokens backwards from end
            break_token_idx = None
            for i in range(end - 1, start - 1, -1):
                token_text = content[token_entries[i][1]:token_entries[i][2]]
                # Check if the text between this token and the next contains a period
                if i + 1 < end:
                    inter_token = content[token_entries[i][2]:token_entries[i + 1][1]]
                else:
                    # Look at text right after the last token in window
                    last_tok_end = token_entries[end - 1][2]
                    next_tok_start = token_entries[end][1] if end < len(token_entries) else last_tok_end + 1
                    inter_token = content[last_tok_end:next_tok_start]

                if '.' in inter_token or '\n' in inter_token:
                    # Prefer breaks past the midpoint
                    tokens_so_far = i - start + 1
                    if tokens_so_far > (end - start) // 2:
                        break_token_idx = i
                        break

            if break_token_idx is not None:
                end = break_token_idx + 1

        # Extract the chunk text from original text using character offsets
        char_start = token_entries[start][1]
        # Include trailing punctuation/whitespace up to the next token (or end of text)
        if end < len(token_entries):
            char_end = token_entries[end][1]  # start of next token (not included)
        else:
            char_end = len(content)

        chunk_text = content[char_start:char_end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Move forward, accounting for overlap
        start = end - overlap
        if start >= end:
            start = end

    return chunks


def load_chunked_documents(folder_path: str, chunk_size: int = 500, overlap: int = 50, max_docs: int = None) -> list:
    """
    Load documents and split them into token-based chunks.

    Args:
        folder_path: Path to folder containing .txt or .md files
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        max_docs: Maximum number of source documents to load (None for unlimited)

    Returns:
        List of dictionaries with 'id', 'content', 'metadata' (one entry per chunk)
    """
    documents = load_documents_from_folder(folder_path, max_docs)
    chunked_docs = []

    for doc in documents:
        chunks = chunk_document(doc['content'], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            # Copy all metadata from source document
            chunk_metadata = doc['metadata'].copy()
            chunk_metadata['chunk'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            # Include dataset in ID to avoid collisions across versions
            dataset = doc['metadata'].get('dataset', '')
            chunked_docs.append({
                'id': f"{dataset}/{doc['id']}#chunk{i}",
                'content': chunk,
                'metadata': chunk_metadata
            })

    return chunked_docs


def get_latest_version(documents: list) -> Optional[str]:
    """
    Extract and compare versions from document metadata to find the latest.

    Args:
        documents: List of documents with metadata

    Returns:
        Latest version string (e.g., 'v53') or None if no versions found
    """
    versions = set()
    for doc in documents:
        version = doc.get('metadata', {}).get('version')
        if version:
            versions.add(version)

    if not versions:
        return None

    def parse_version(v: str):
        """Parse version string to comparable tuple. e.g., 'v0.52.0' -> (0, 52, 0)"""
        if not v.startswith('v'):
            return (0,)
        parts = v[1:].split('.')
        return tuple(int(p) for p in parts)

    latest = max(versions, key=parse_version)
    return latest


def detect_version_in_query(query: str, documents: list) -> Optional[str]:
    """
    Detect if the user query mentions a specific version.

    Handles patterns like: v52, 52, version 52, ver 52, hyprland 42, etc.

    Args:
        query: User query string
        documents: List of documents with metadata (to extract available versions)

    Returns:
        Detected version string (e.g., 'v52') or None if no version detected
    """
    import re

    # Get all available versions
    versions = set()
    for doc in documents:
        version = doc.get('metadata', {}).get('version')
        if version:
            versions.add(version)

    if not versions:
        return None

    # Build a set of version numbers (without 'v' prefix) for matching bare numbers
    version_numbers = set()
    for v in versions:
        norm = normalize_version(v)
        # Extract numeric part: 'v52' -> '52', 'v0.52.0' -> '0.52.0'
        if norm.startswith('v'):
            version_numbers.add(norm[1:])

    # Pattern to match version-like strings in query
    version_patterns = [
        r'\bv(\d+(?:\.\d+)*)\b',  # v52, v0.52, v0.52.0
        r'\bversion\s*(\d+(?:\.\d+)*)\b',  # version 52, version 0.52
        r'\bver\s*(\d+(?:\.\d+)*)\b',  # ver 52
    ]

    for pattern in version_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            detected = normalize_version(match)
            for version in versions:
                norm_version = normalize_version(version)
                if detected == norm_version or detected in norm_version or norm_version in detected:
                    return version

    # Fallback: match bare numbers that correspond to known versions
    # e.g., "hyprland 42" -> detect v42
    bare_number_pattern = r'\b(\d+(?:\.\d+)*)\b'
    bare_matches = re.findall(bare_number_pattern, query)
    for match in bare_matches:
        if match in version_numbers:
            # Find the corresponding version with 'v' prefix
            for version in versions:
                norm = normalize_version(version)
                if norm.startswith('v') and norm[1:] == match:
                    return version

    return None
