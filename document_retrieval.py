"""
Document Retrieval Module
Implements intelligent document chunking and retrieval strategies
"""

import re
from typing import List, Tuple, Dict


def chunk_document(document: str, chunk_size: int = 1500, overlap: int = 300) -> List[Dict]:
    """
    Split a document into overlapping chunks for better context retrieval.
    
    Args:
        document: Full text of the document
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
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
            
            # Overlap: include last few sentences in next chunk
            overlap_text = " ".join(current_chunk.split()[-3:])  # Last 3 words
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


def create_chunked_documents(documents: List[str], doc_ids: List[str], doc_metadata: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Create chunks for all documents and maintain mapping to original documents.
    
    Returns:
        Tuple of (chunks_with_metadata, doc_id_mapping)
        - chunks_with_metadata: List of chunk dicts with original doc info
        - doc_id_mapping: List mapping chunk index to original doc index
    """
    chunks_with_metadata = []
    doc_id_mapping = []
    
    for doc_idx, (document, doc_id, metadata) in enumerate(zip(documents, doc_ids, doc_metadata)):
        chunks = chunk_document(document)
        
        for chunk in chunks:
            chunk_data = {
                'text': chunk['text'],
                'doc_id': doc_id,
                'doc_title': metadata.get('title', 'Untitled'),
                'doc_type': metadata.get('itemType', 'Unknown'),
                'doc_abstract': metadata.get('abstract', ''),
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
    k: int = 5,
    similarity_threshold: float = 0.05
) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query using TF-IDF scoring.
    
    Args:
        query: User's question
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: TF-IDF matrix of chunk texts
        chunks_with_metadata: List of chunk metadata
        doc_id_mapping: Mapping from chunk index to doc index
        k: Number of top chunks to retrieve
        similarity_threshold: Minimum similarity score to include
    
    Returns:
        List of relevant chunks sorted by similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    relevant_chunks = []
    seen_docs = {}  # Track docs we've already cited
    
    for idx in top_indices:
        if similarities[idx] > similarity_threshold:
            chunk = chunks_with_metadata[idx].copy()
            chunk['similarity'] = float(similarities[idx])
            chunk['original_doc_index'] = doc_id_mapping[idx]
            
            relevant_chunks.append(chunk)
            
            # Track which docs we're citing
            doc_id = chunk['doc_id']
            if doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    'title': chunk['doc_title'],
                    'type': chunk['doc_type'],
                    'max_similarity': chunk['similarity'],
                    'chunk_count': 0
                }
            seen_docs[doc_id]['chunk_count'] += 1
            seen_docs[doc_id]['max_similarity'] = max(seen_docs[doc_id]['max_similarity'], chunk['similarity'])
    
    return relevant_chunks, seen_docs


def format_context_from_chunks(relevant_chunks: List[Dict], seen_docs: Dict) -> Tuple[str, List[Dict]]:
    """
    Format retrieved chunks into context string for the LLM.
    
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
            context_parts.append(f"\n[Section {i}]\n{chunk['text']}")
    
    context = "\n".join(context_parts)
    
    # Create citations list
    cited_docs = [
        {
            'title': info['title'],
            'id': doc_id,
            'similarity': info['max_similarity'],
            'chunk_count': info['chunk_count']
        }
        for doc_id, info in seen_docs.items()
    ]
    
    return context, cited_docs
