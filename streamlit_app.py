import streamlit as st
from pyzotero import zotero
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import PyPDF2
import requests
import io
import json
from streamlit_extras.bottom_container import bottom

from article_storage import (
    load_articles, add_article, get_all_articles, remove_article,
    clear_all_articles, rename_article, article_exists
)
from document_retrieval import (
    create_chunked_documents,
    retrieve_relevant_chunks,
    format_context_from_chunks,
    clear_chunk_cache,
    load_chunk_index,      # for reference
    save_chunk_index       # for reference
)


# ==================== CONFIG ====================
COLLECTIONS = [
    {"name": "Agile BioFoundry", "collection_key": "agile"},
    {"name": "ABPDU", "collection_key": "abpdu"},
]

# ==================== HELPERS ====================
def extract_pdf_text(pdf_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for i, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            text += f"Page {i}\n" + page_text + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"


def _safe_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return default


def get_collection_zotero_key(collection_key: str) -> str:
    return _safe_secret(f"zotero_collection_{collection_key}")


def is_item_low_relevance(item: dict) -> bool:
    tags = item.get('data', {}).get('tags', []) or []
    for tag in tags:
        tag_value = tag.get('tag') if isinstance(tag, dict) else str(tag)
        normalized = tag_value.strip().lower().replace('_', ' ').replace('-', ' ')
        if normalized in ['low relevance', 'low-relevance', 'low_relevance', 'lowrelevance']:
            return True
        if 'low' in normalized and 'relevance' in normalized:
            return True
    return False


def init_session_state_for_collection(collection_key: str):
    prefix = f"col_{collection_key}"
    if f"{prefix}_documents" not in st.session_state:
        documents, doc_ids, doc_metadata = get_all_articles(collection_key)
        st.session_state[f"{prefix}_documents"] = documents
        st.session_state[f"{prefix}_doc_ids"] = doc_ids
        st.session_state[f"{prefix}_doc_metadata"] = doc_metadata


def ensure_chunk_index_for_collection(collection_key: str) -> bool:
    """Load from cache or build + persist new index."""
    prefix = f"col_{collection_key}"

    # Try cache first
    loaded = load_chunk_index(collection_key)
    if loaded:
        chunks, doc_id_mapping, vectorizer, faiss_index = loaded
        st.session_state[f"{prefix}_chunks"] = chunks
        st.session_state[f"{prefix}_doc_id_mapping"] = doc_id_mapping
        st.session_state[f"{prefix}_chunk_vectorizer"] = vectorizer
        st.session_state[f"{prefix}_faiss_index"] = faiss_index
        return True

    # Build fresh
    if not st.session_state.get(f'{prefix}_documents'):
        return False

    chunks, doc_id_mapping = create_chunked_documents(
        st.session_state[f'{prefix}_documents'],
        st.session_state[f'{prefix}_doc_ids'],
        st.session_state[f'{prefix}_doc_metadata'],
        chunk_size=st.session_state.get('chunk_size', 500),
        overlap=st.session_state.get('overlap', 80),
        summary_sentences=st.session_state.get('summary_sentences', 1)
    )

    if not chunks:
        return False

    st.session_state[f"{prefix}_chunks"] = chunks
    st.session_state[f"{prefix}_doc_id_mapping"] = doc_id_mapping

    chunk_texts = [c.get('summary', c['text']) for c in chunks]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    chunk_vectors = tfidf_matrix.toarray().astype('float32')
    faiss.normalize_L2(chunk_vectors)
    faiss_index = faiss.IndexFlatIP(chunk_vectors.shape[1])
    faiss_index.add(chunk_vectors)

    st.session_state[f"{prefix}_chunk_vectorizer"] = vectorizer
    st.session_state[f"{prefix}_faiss_index"] = faiss_index

    # Persist
    save_chunk_index(chunks, doc_id_mapping, vectorizer, faiss_index, collection_key)
    return True


# ==================== MAIN APP ====================
st.set_page_config(page_title="Agile Biofoundry & ABPDU Query Tool", layout="wide")
st.title("Agile Biofoundry & ABPDU Query Tool")

zotero_library_id = _safe_secret("zotero_library_id")
zotero_api_key = _safe_secret("zotero_api_key")
zotero_library_type = _safe_secret("zotero_library_type", "user")
openai_api_key = _safe_secret("openai_api_key")

client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Global settings optimized for long content
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 500
if 'overlap' not in st.session_state:
    st.session_state.overlap = 80
if 'summary_sentences' not in st.session_state:
    st.session_state.summary_sentences = 1
if 'use_summaries' not in st.session_state:
    st.session_state.use_summaries = True
if 'k_chunks' not in st.session_state:
    st.session_state.k_chunks = 3
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}

for col_info in COLLECTIONS:
    init_session_state_for_collection(col_info['collection_key'])

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Collections & Documents")
    for col_info in COLLECTIONS:
        collection_key = col_info["collection_key"]
        collection_name = col_info["name"]
        prefix = f"col_{collection_key}"
        
        with st.expander(f"{collection_name}", expanded=False):
            st.metric("Documents", len(st.session_state.get(f"{prefix}_documents", [])))
            
            # ... (your existing search, document list, rename/delete logic) ...
            # Important: After any mutation (delete, rename, clear), call:
            # clear_chunk_cache(collection_key)
            # ensure_chunk_index_for_collection(collection_key)
            
            # Example for delete (adapt to your popover code):
            # after remove_article():
            # clear_chunk_cache(collection_key)
            # ensure_chunk_index_for_collection(collection_key)

# ==================== TABS & CHAT ====================
tabs = st.tabs([col["name"] for col in COLLECTIONS])

for tab, col_info in zip(tabs, COLLECTIONS):
    with tab:
        collection_key = col_info["collection_key"]
        prefix = f"col_{collection_key}"
        
        st.header(f"{col_info['name']} Collection")
        
        # Load from Zotero button (your existing code)
        # After successful load and updating session state:
        # clear_chunk_cache(collection_key)
        # ensure_chunk_index_for_collection(collection_key)
        
        # Chat interface
        if f"{prefix}_messages" not in st.session_state:
            st.session_state[f"{prefix}_messages"] = []
        
        for message in st.session_state[f"{prefix}_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Processing pending prompt (key optimization)
        if st.session_state.get(f'{prefix}_processing') and st.session_state.get(f'{prefix}_pending_prompt'):
            pending = st.session_state[f'{prefix}_pending_prompt']
            try:
                with st.spinner("Thinking..."):
                    cache_key = f"{collection_key}:{pending.lower().strip()}"
                    if cache_key in st.session_state.query_cache:
                        result = st.session_state.query_cache[cache_key]
                    else:
                        if not ensure_chunk_index_for_collection(collection_key):
                            result = "No indexed documents available."
                        else:
                            relevant_chunks, seen_docs = retrieve_relevant_chunks(
                                pending,
                                st.session_state[f"{prefix}_chunk_vectorizer"],
                                st.session_state[f"{prefix}_chunks"],
                                st.session_state[f"{prefix}_doc_id_mapping"],
                                faiss_index=st.session_state.get(f"{prefix}_faiss_index"),
                                k=st.session_state.get('k_chunks', 3),
                                similarity_threshold=0.12
                            )

                            context, cited_docs = format_context_from_chunks(
                                relevant_chunks, seen_docs, use_summaries=st.session_state.get('use_summaries', True)
                            )

                            if client is None:
                                result = "OpenAI API key not configured."
                            else:
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": f"You are a helpful assistant knowledgeable about {col_info['name']}."},
                                        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {pending}"}
                                    ],
                                    stream=True
                                )
                                result = "".join(chunk.choices[0].delta.content or "" for chunk in response if chunk.choices[0].delta.content)

                                if cited_docs:
                                    result += "\n\n---\n**Sources:**\n"
                                    for doc in cited_docs:
                                        extra = []
                                        if doc.get('pages'):
                                            extra.append("pages " + ",".join(map(str, doc['pages'])))
                                        if doc.get('timestamps'):
                                            extra.append("times " + ",".join(doc['timestamps']))
                                        extras = f" ({'; '.join(extra)})" if extra else ""
                                        result += f"- {doc['title']}{extras} (ID: {doc['id']}, Relevance: {doc['similarity']:.2%})\n"
                    
                    st.session_state.query_cache[cache_key] = result
            except Exception as e:
                result = f"Error: {str(e)}"

            # Update assistant message
            assistant_idx = st.session_state.get(f'{prefix}_pending_assistant_index')
            if assistant_idx is not None:
                st.session_state[f"{prefix}_messages"][assistant_idx]['content'] = result
            st.session_state[f'{prefix}_pending_prompt'] = None
            st.session_state[f'{prefix}_processing'] = False
            st.rerun()

        # Chat input
        if prompt := st.chat_input(f"Ask about {col_info['name']}:", key=f"chat_input_{collection_key}"):
            st.session_state[f"{prefix}_messages"].append({"role": "user", "content": prompt})
            st.session_state[f"{prefix}_messages"].append({"role": "assistant", "content": "Thinking..."})
            st.session_state[f'{prefix}_pending_prompt'] = prompt
            st.session_state[f'{prefix}_pending_assistant_index'] = len(st.session_state[f"{prefix}_messages"]) - 1
            st.session_state[f'{prefix}_processing'] = True
            st.rerun()

with bottom():
    st.write("Zotero Library Source: https://www.zotero.org/groups/6420515/abpdu_workflow_automation-article_query_tool/collections/LRILZKMS/collection")