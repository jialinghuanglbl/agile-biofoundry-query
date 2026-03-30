"""
Document Retrieval Module
Implements intelligent document chunking and retrieval strategies
"""

import re
from typing import List, Tuple, Dict


def chunk_document(document: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """
    Split a document into overlapping chunks for better context retrieval.
    
    Args:
        document: Full text of the document
        chunk_size: Target size of each chunk in characters
        overlap: Target overlap in characters (approximate, converted to words)
    
    Returns:
        List of chunk dictionaries with text and position info
    """
    # Try to split on sentences first for coherence
    sentences = re.split(r'(?<=[.!?])\s+', document)
    
    chunks = []
    current_chunk = ""
    chunk_start_idx = 0
    
    for sentence in sentences:
        # Check if adding this sentence would exceed chunk size
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if len(potential_chunk) > chunk_size and current_chunk:
            # Save current chunk and start new one
            chunks.append({
                'text': current_chunk.strip(),
                'start': chunk_start_idx,
                'end': chunk_start_idx + len(current_chunk)
            })
            
            # Compute approximate overlap in words (to avoid chopping words/sentences)
            approx_words = max(1, overlap // 5)  # assume ~5 chars/word
            overlap_text = " ".join(current_chunk.split()[-approx_words:])
            current_chunk = overlap_text + " " + sentence
            chunk_start_idx = chunk_start_idx + len(current_chunk) - len(overlap_text)
        else:
            current_chunk = potential_chunk
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'start': chunk_start_idx,
            'end': chunk_start_idx + len(current_chunk)
        })
    
    return chunks


def is_transcript_doc(document: str, metadata: Dict) -> bool:
    """Heuristic to detect if a document is a transcript (e.g., YouTube).

    Checks title/abstract for keywords, looks for timestamp patterns, and
    inspects line lengths to decide.
    """
    title = metadata.get('title', '').lower() if metadata else ''
    abstract = metadata.get('abstract', '').lower() if metadata else ''

    if 'transcript' in title or 'transcript' in abstract:
        return True
    if 'youtube' in title or 'youtube' in abstract:
        return True

    # Timestamp patterns like 00:01 or 1:23:45
    if re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", document):
        return True

    # If the document has many short lines, it's likely a transcript
    lines = [ln.strip() for ln in document.splitlines() if ln.strip()]
    if len(lines) >= 10:
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        if avg_len < 100:
            return True

    return False


def chunk_transcript(document: str, chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
    """Chunk a transcript-like document by fixed character windows after cleanup."""
    # Remove obvious timestamps
    doc = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", document)
    doc = re.sub(r"\s+", " ", doc).strip()

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(doc), step):
        text = doc[start:start + chunk_size].strip()
        if not text:
            continue
        chunks.append({
            'text': text,
            'start': start,
            'end': start + len(text)
        })

    return chunks


def summarize_chunk(text: str, max_sentences: int = 2) -> str:
    """
    Simple extractive summarization: pick the top scoring sentences by TF-IDF.
    This is fast and keeps important sentences while reducing token usage.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return text.strip()

    try:
        vec = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vec.fit_transform(sentences)
        scores = X.sum(axis=1).A1
        top_idx = np.argsort(scores)[-max_sentences:]
        top_idx_sorted = sorted(top_idx)
        summary = " ".join([sentences[i].strip() for i in top_idx_sorted])
        return summary
    except Exception:
        # Fallback: return the first few sentences
        return " ".join(sentences[:max_sentences]).strip()


def create_chunked_documents(
    documents: List[str],
    doc_ids: List[str],
    doc_metadata: List[Dict],
    chunk_size: int = 800,
    overlap: int = 150,
    summary_sentences: int = 3
) -> Tuple[List[Dict], List[str]]:
    """
    Create chunks for all documents and maintain mapping to original documents.
    Supports summarization of chunks to reduce token usage during retrieval.
    
    Returns:
        Tuple of (chunks_with_metadata, doc_id_mapping)
        - chunks_with_metadata: List of chunk dicts with original doc info
        - doc_id_mapping: List mapping chunk index to original doc index
    """
    chunks_with_metadata = []
    doc_id_mapping = []

    for doc_idx, (document, doc_id, metadata) in enumerate(zip(documents, doc_ids, doc_metadata)):
        # If this looks like a transcript (YouTube/video), use narrower, fixed-size chunking
        if is_transcript_doc(document, metadata):
            # narrower chunking and smaller summaries for transcripts
            transcript_chunk_size = min(400, chunk_size)
            transcript_overlap = min(80, overlap)
            chunks = chunk_transcript(document, chunk_size=transcript_chunk_size, overlap=transcript_overlap)
            use_summary_sentences = max(1, summary_sentences - 1)
        else:
            chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
            use_summary_sentences = summary_sentences

        for chunk in chunks:
            summary = summarize_chunk(chunk['text'], max_sentences=use_summary_sentences)
            chunk_data = {
                'text': chunk['text'],
                'summary': summary,
                'doc_id': doc_id,
                'doc_title': metadata.get('title', 'Untitled'),
                'doc_type': metadata.get('itemType', 'Unknown'),
                'doc_abstract': metadata.get('abstract', ''),
                'low_relevance': metadata.get('low_relevance', False),
                'chunk_position': len([c for c in chunks_with_metadata if c['doc_id'] == doc_id])
            }
            chunks_with_metadata.append(chunk_data)
            doc_id_mapping.append(doc_idx)

    return chunks_with_metadata, doc_id_mapping


def retrieve_relevant_chunks(
    query: str,
    vectorizer,
    tfidf_matrix,
    chunks_with_metadata: List[Dict],
    doc_id_mapping: List[int],
    faiss_index=None,
    k: int = 5,
    similarity_threshold: float = 0.05
) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query using TF-IDF scoring with ANN approximation.
    
    Args:
        query: User's question
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: TF-IDF matrix of chunk texts
        chunks_with_metadata: List of chunk metadata
        doc_id_mapping: Mapping from chunk index to doc index
        faiss_index: Pre-built FAISS index (optional, falls back to exact if None)
        k: Number of top chunks to retrieve
        similarity_threshold: Minimum similarity score to include
    
    Returns:
        List of relevant chunks sorted by similarity
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    
    if faiss_index is not None:
        # Use ANN search
        query_vec = vectorizer.transform([query]).toarray().astype('float32')
        
        # ANN search: get more candidates than needed for better recall
        search_k = min(max(k * 5, 100), len(chunks_with_metadata))
        similarities, indices = faiss_index.search(query_vec, search_k)
        
        # Apply low-relevance weighting in parallel
        low_relevance_weight = 0.425
        def adjust_score(i):
            raw_score = float(similarities[0][i])
            chunk = chunks_with_metadata[indices[0][i]]
            return raw_score * low_relevance_weight if chunk.get('low_relevance') else raw_score
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            adjusted_similarities = list(executor.map(adjust_score, range(len(indices[0]))))
        
        # Sort by adjusted similarity
        sorted_pairs = sorted(zip(adjusted_similarities, indices[0]), reverse=True)
        candidate_indices = [idx for _, idx in sorted_pairs]
    else:
        raise ValueError("FAISS index is required for retrieval")
    
    relevant_chunks = []
    seen_docs = {}  # Track docs we've already cited
    doc_ids_included = set()
    min_source_docs = 5
    
    for idx in candidate_indices:
        adjusted_score = adjusted_similarities[candidate_indices.index(idx)]
        chunk = chunks_with_metadata[idx].copy()
        doc_id = chunk['doc_id']
    
        should_include = adjusted_score > similarity_threshold or len(doc_ids_included) < min_source_docs
        if not should_include:
            continue
    
        chunk['similarity'] = float(adjusted_score)
        chunk['original_doc_index'] = doc_id_mapping[idx]
    
        doc_type = chunk.get('doc_type', '').lower()
        # decide what to annotate: articles (pdf/text) get pages; videos/audio (transcripts) get timestamps
        if doc_type in ['videorecording', 'audiorecording'] or 'transcript' in doc_type:
            # timestamp detection only
            ts_match = re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", chunk['text'])
            chunk['page'] = None
            chunk['timestamp'] = ts_match.group(0) if ts_match else None
        else:
            # page-number detection only
            page_match = re.search(r"Page\s+(\d+)", chunk['text'], re.IGNORECASE)
            chunk['page'] = page_match.group(1) if page_match else None
            chunk['timestamp'] = None
    
        relevant_chunks.append(chunk)
        doc_ids_included.add(doc_id)
    
        if doc_id not in seen_docs:
            seen_docs[doc_id] = {
                'title': chunk['doc_title'],
                'type': chunk['doc_type'],
                'max_similarity': chunk['similarity'],
                'chunk_count': 0,
                'pages': set(),
                'timestamps': set()
            }
    
        seen_docs[doc_id]['chunk_count'] += 1
        seen_docs[doc_id]['max_similarity'] = max(seen_docs[doc_id]['max_similarity'], chunk['similarity'])
        if chunk.get('page'):
            seen_docs[doc_id]['pages'].add(chunk['page'])
        if chunk.get('timestamp'):
            seen_docs[doc_id]['timestamps'].add(chunk['timestamp'])
    
        if len(doc_ids_included) >= min_source_docs and len(relevant_chunks) >= k:
            break
    
    # Ensure we still return a top-k chunk set even if the source minimum wasn't reached (e.g., small dataset)
    if len(relevant_chunks) > k:
        relevant_chunks = relevant_chunks[:k]
    
    return relevant_chunks, seen_docs


def format_context_from_chunks(
    relevant_chunks: List[Dict],
    seen_docs: Dict,
    use_summaries: bool = True
) -> Tuple[str, List[Dict]]:
    """
    Format retrieved chunks into context string for the LLM.

    If `use_summaries` is True, include chunk summaries (shorter). Otherwise include full chunk text.
    Returns:
        Tuple of (formatted_context, cited_docs_list)
    """
    context_parts = []

    # Group chunks by document for better readability
    chunks_by_doc = {}
    for chunk in relevant_chunks:
        doc_id = chunk['doc_id']
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)

    # Format context by document
    for doc_id, chunks in chunks_by_doc.items():
        title = chunks[0]['doc_title']
        context_parts.append(f"\n**Source: {title}**")

        for i, chunk in enumerate(chunks, 1):
            text_to_include = chunk.get('summary') if use_summaries and chunk.get('summary') else chunk['text']
            notes = []
            if chunk.get('page'):
                notes.append(f"page {chunk['page']}")
            if chunk.get('timestamp'):
                notes.append(f"~{chunk['timestamp']}")
            section_label = f"[Section {i}"
            if notes:
                section_label += " (" + ", ".join(notes) + ")"
            section_label += "]"
            context_parts.append(f"\n{section_label}\n{text_to_include}")

    context = "\n".join(context_parts)

    # Create citations list
    cited_docs = []
    for doc_id, info in seen_docs.items():
        entry = {
            'title': info['title'],
            'id': doc_id,
            'similarity': info['max_similarity'],
            'chunk_count': info['chunk_count']
        }
        if info.get('pages'):
            entry['pages'] = sorted(info['pages'], key=lambda x: int(x))
        if info.get('timestamps'):
            entry['timestamps'] = sorted(info['timestamps'])
        cited_docs.append(entry)

    return context, cited_docs
