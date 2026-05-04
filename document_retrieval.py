"""
Document Retrieval Module
"""

import re
import os
import joblib
import shutil
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    print("WARNING: faiss not installed. Run: pip install faiss-cpu")


def _get_cache_dir(collection_key: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "zotero_cache", collection_key)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def clear_chunk_cache(collection_key: str) -> None:
    """Clear persistent cache when documents change."""
    cache_dir = _get_cache_dir(collection_key)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def save_chunk_index(
    chunks: List[Dict],
    doc_id_mapping: List[int],
    vectorizer,
    faiss_index,
    collection_key: str
) -> None:
    if not FAISS_AVAILABLE or faiss_index is None:
        return
    cache_dir = _get_cache_dir(collection_key)
    joblib.dump(chunks, os.path.join(cache_dir, "chunks.joblib"))
    joblib.dump(doc_id_mapping, os.path.join(cache_dir, "doc_id_mapping.joblib"))
    joblib.dump(vectorizer, os.path.join(cache_dir, "chunk_vectorizer.joblib"))
    faiss.write_index(faiss_index, os.path.join(cache_dir, "faiss_index.bin"))


def load_chunk_index(collection_key: str) -> Optional[Tuple[List[Dict], List[int], TfidfVectorizer, "faiss.Index"]]:
    if not FAISS_AVAILABLE:
        return None
    cache_dir = _get_cache_dir(collection_key)
    chunks_path = os.path.join(cache_dir, "chunks.joblib")
    if not os.path.exists(chunks_path):
        return None

    try:
        chunks = joblib.load(chunks_path)
        doc_id_mapping = joblib.load(os.path.join(cache_dir, "doc_id_mapping.joblib"))
        vectorizer = joblib.load(os.path.join(cache_dir, "chunk_vectorizer.joblib"))
        faiss_index = faiss.read_index(os.path.join(cache_dir, "faiss_index.bin"))
        return chunks, doc_id_mapping, vectorizer, faiss_index
    except Exception:
        return None


# ==================== CHUNKING ====================
def chunk_document(document: str, chunk_size: int = 500, overlap: int = 80) -> List[Dict]:
    sentences = re.split(r'(?<=[.!?])\s+', document)
    chunks = []
    current_chunk = ""
    chunk_start_idx = 0

    for sentence in sentences:
        potential = current_chunk + " " + sentence if current_chunk else sentence
        if len(potential) > chunk_size and current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start': chunk_start_idx,
                'end': chunk_start_idx + len(current_chunk)
            })
            approx_words = max(1, overlap // 5)
            overlap_text = " ".join(current_chunk.split()[-approx_words:])
            current_chunk = overlap_text + " " + sentence
            chunk_start_idx += len(current_chunk) - len(overlap_text)
        else:
            current_chunk = potential

    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'start': chunk_start_idx,
            'end': chunk_start_idx + len(current_chunk)
        })
    return chunks


def is_transcript_doc(document: str, metadata: Dict) -> bool:
    title = metadata.get('title', '').lower()
    abstract = metadata.get('abstract', '').lower()

    if any(kw in (title + abstract) for kw in ['transcript', 'youtube', 'video', 'audio']):
        return True
    if re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", document):
        return True

    lines = [ln.strip() for ln in document.splitlines() if ln.strip()]
    if len(lines) >= 10 and sum(len(ln) for ln in lines) / len(lines) < 100:
        return True

    if len(document) > 30000 or len(lines) > 400:
        return True
    return False


def chunk_transcript(document: str, chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
    doc = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", document)
    doc = re.sub(r"\s+", " ", doc).strip()
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(doc), step):
        text = doc[start:start + chunk_size].strip()
        if text:
            chunks.append({'text': text, 'start': start, 'end': start + len(text)})
    return chunks


def summarize_chunk(text: str, max_sentences: int = 1, _vectorizer: TfidfVectorizer = None) -> str:
    """
    Summarize a chunk to its most informative sentence.
    Accepts an optional pre-fitted vectorizer to avoid re-fitting per chunk.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return text.strip()

    try:
        if _vectorizer is not None:
            X = _vectorizer.transform(sentences)
        else:
            vec = TfidfVectorizer(stop_words='english', max_features=1000)
            X = vec.fit_transform(sentences)
        scores = X.sum(axis=1).A1
        top_idx = np.argsort(scores)[-max_sentences:]
        return " ".join(sentences[i].strip() for i in sorted(top_idx))
    except Exception:
        return " ".join(sentences[:max_sentences]).strip()


def create_chunked_documents(
    documents: List[str],
    doc_ids: List[str],
    doc_metadata: List[Dict],
    chunk_size: int = 500,
    overlap: int = 80,
    summary_sentences: int = 1
) -> Tuple[List[Dict], List[int]]:
    """
    Chunk all documents and compute per-chunk summaries.
    Uses a single shared TF-IDF vectorizer fitted on all chunk texts
    to avoid the cost of re-fitting a vectorizer for every individual chunk.
    """
    # First pass: collect all raw chunks
    raw_chunks = []
    raw_doc_id_mapping = []

    for doc_idx, (document, doc_id, metadata) in enumerate(zip(documents, doc_ids, doc_metadata)):
        if is_transcript_doc(document, metadata):
            chunks = chunk_transcript(document, min(400, chunk_size), min(80, overlap))
            use_summary_sentences = 1
        else:
            chunks = chunk_document(document, chunk_size, overlap)
            use_summary_sentences = summary_sentences

        for chunk in chunks:
            raw_chunks.append({
                'text': chunk['text'],
                'doc_id': doc_id,
                'doc_title': metadata.get('title', 'Untitled'),
                'doc_type': metadata.get('itemType', 'Unknown'),
                'doc_abstract': metadata.get('abstract', ''),
                'doc_url': metadata.get('url', ''),
                'doc_image_urls': metadata.get('image_urls', []),
                'low_relevance': metadata.get('low_relevance', False),
                'use_summary_sentences': use_summary_sentences,
            })
            raw_doc_id_mapping.append(doc_idx)

    if not raw_chunks:
        return [], []

    # Fit a single vectorizer across all chunk texts for summarization
    all_texts = [c['text'] for c in raw_chunks]
    try:
        shared_vec = TfidfVectorizer(stop_words='english', max_features=1000)
        shared_vec.fit(all_texts)
    except Exception:
        shared_vec = None

    # Second pass: compute summaries using the shared vectorizer
    chunks_with_metadata = []
    doc_chunk_counts: Dict[str, int] = {}

    for chunk_data in raw_chunks:
        doc_id = chunk_data['doc_id']
        doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0)

        summary = summarize_chunk(
            chunk_data['text'],
            chunk_data['use_summary_sentences'],
            _vectorizer=shared_vec
        )

        chunks_with_metadata.append({
            'text': chunk_data['text'],
            'summary': summary,
            'doc_id': doc_id,
            'doc_title': chunk_data['doc_title'],
            'doc_type': chunk_data['doc_type'],
            'doc_abstract': chunk_data['doc_abstract'],
            'doc_url': chunk_data.get('doc_url', ''),
            'low_relevance': chunk_data['low_relevance'],
            'chunk_position': doc_chunk_counts[doc_id],
        })
        doc_chunk_counts[doc_id] += 1

    return chunks_with_metadata, raw_doc_id_mapping


def retrieve_relevant_chunks(
    query: str,
    vectorizer,
    chunks_with_metadata: List[Dict],
    doc_id_mapping: List[int],
    faiss_index=None,
    k: int = 3,
    similarity_threshold: float = 0.12
) -> Tuple[List[Dict], Dict]:
    if not FAISS_AVAILABLE or faiss_index is None:
        raise ImportError("FAISS is required but not available. Make sure faiss-cpu is installed.")

    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    faiss.normalize_L2(query_vec)

    ann_k = min(k * 4, len(chunks_with_metadata))
    similarities, indices = faiss_index.search(query_vec, ann_k)
    similarities = similarities.flatten()
    indices = indices.flatten()

    low_relevance_weight = 0.425
    adjusted = []
    for i, idx in enumerate(indices):
        chunk = chunks_with_metadata[idx]
        adj_sim = similarities[i] * low_relevance_weight if chunk.get('low_relevance') else similarities[i]
        adjusted.append((idx, adj_sim))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    candidate_indices = [idx for idx, _ in adjusted[:ann_k]]

    relevant_chunks = []
    seen_docs = {}
    doc_ids_included = set()

    for idx in candidate_indices:
        adj_score = next((s for i, s in adjusted if i == idx), 0)
        if adj_score <= similarity_threshold and len(doc_ids_included) >= 5:
            continue

        chunk = chunks_with_metadata[idx].copy()
        doc_id = chunk['doc_id']
        chunk['similarity'] = float(adj_score)
        chunk['original_doc_index'] = doc_id_mapping[idx]

        doc_type = chunk.get('doc_type', '').lower()
        if doc_type in ['videorecording', 'audiorecording'] or 'transcript' in doc_type:
            ts = re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", chunk['text'])
            chunk['timestamp'] = ts.group(0) if ts else None
            chunk['page'] = None
        else:
            pg = re.search(r"Page\s+(\d+)", chunk['text'], re.IGNORECASE)
            chunk['page'] = pg.group(1) if pg else None
            chunk['timestamp'] = None

        relevant_chunks.append(chunk)
        doc_ids_included.add(doc_id)

        if doc_id not in seen_docs:
            seen_docs[doc_id] = {
                'title': chunk['doc_title'],
                'type': chunk['doc_type'],
                'url': chunk.get('doc_url', ''),
                'image_urls': chunk.get('doc_image_urls', []),
                'max_similarity': chunk['similarity'],
                'chunk_count': 0,
                'pages': set(),
                'timestamps': set()
            }
        info = seen_docs[doc_id]
        info['chunk_count'] += 1
        info['max_similarity'] = max(info.get('max_similarity', 0), chunk['similarity'])
        if chunk.get('page'):
            info['pages'].add(chunk['page'])
        if chunk.get('timestamp'):
            info['timestamps'].add(chunk['timestamp'])

        if len(doc_ids_included) >= 5 and len(relevant_chunks) >= k:
            break

    if len(relevant_chunks) > k:
        relevant_chunks = relevant_chunks[:k]

    return relevant_chunks, seen_docs


def format_context_from_chunks(
    relevant_chunks: List[Dict],
    seen_docs: Dict,
    use_summaries: bool = True
) -> Tuple[str, List[Dict]]:
    context_parts = []
    chunks_by_doc = {}
    for chunk in relevant_chunks:
        chunks_by_doc.setdefault(chunk['doc_id'], []).append(chunk)

    for doc_id, chunks in chunks_by_doc.items():
        title = chunks[0]['doc_title']
        context_parts.append(f"\n**Source: {title}**")
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('summary') if use_summaries and chunk.get('summary') else chunk['text']
            notes = []
            if chunk.get('page'):
                notes.append(f"page {chunk['page']}")
            if chunk.get('timestamp'):
                notes.append(f"~{chunk['timestamp']}")
            section = f"[Section {i}"
            if notes:
                section += f" ({', '.join(notes)})"
            section += "]"
            context_parts.append(f"\n{section}\n{text}")

    context = "\n".join(context_parts)

    cited_docs = []
    for doc_id, info in seen_docs.items():
        entry = {
            'title': info['title'],
            'id': doc_id,
            'url': info.get('url', ''),
            'image_urls': info.get('image_urls', []),
            'similarity': info['max_similarity'],
            'chunk_count': info['chunk_count']
        }
        if info.get('pages'):
            entry['pages'] = sorted(info['pages'], key=int)
        if info.get('timestamps'):
            entry['timestamps'] = sorted(info['timestamps'])
        cited_docs.append(entry)

    return context, cited_docs