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
    load_chunk_index,
    save_chunk_index
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

# Optimized defaults for long documents
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
            
            # Search
            search_term = st.text_input(f"Search {collection_name}", key=f"search_{collection_key}", placeholder="Search documents...")
            
            # Document list with actions
            docs = st.session_state.get(f"{prefix}_doc_metadata", [])
            filtered_docs = [i for i, m in enumerate(docs) if not search_term or search_term.lower() in m.get('title', '').lower()]
            
            for idx in filtered_docs:
                meta = docs[idx]
                title_short = meta.get('title', 'Untitled')[:50]
                with st.popover(title_short, use_container_width=True):
                    full_title = meta.get('title', 'Untitled')
                    zotero_id = st.session_state[f"{prefix}_doc_ids"][idx]
                    st.write(f"**{full_title}**")
                    st.caption(f"ID: {zotero_id}")
                    
                    # Rename
                    new_title = st.text_input("New title", value=full_title, key=f"rename_in_{collection_key}_{idx}")
                    if st.button("Rename", key=f"rename_btn_{collection_key}_{idx}"):
                        success, msg = rename_article(zotero_id, new_title, collection_key)
                        if success:
                            st.session_state[f"{prefix}_doc_metadata"][idx]['title'] = new_title
                            clear_chunk_cache(collection_key)
                            ensure_chunk_index_for_collection(collection_key)
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    
                    # Delete
                    if st.button("🗑️ Delete", key=f"delete_btn_{collection_key}_{idx}", type="secondary"):
                        remove_article(zotero_id, collection_key)
                        del st.session_state[f"{prefix}_documents"][idx]
                        del st.session_state[f"{prefix}_doc_ids"][idx]
                        del st.session_state[f"{prefix}_doc_metadata"][idx]
                        clear_chunk_cache(collection_key)
                        ensure_chunk_index_for_collection(collection_key)
                        st.success("Document deleted")
                        st.rerun()
            
            # Bulk actions
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Clear All", key=f"clear_all_{collection_key}", use_container_width=True):
                    clear_all_articles(collection_key)
                    st.session_state[f"{prefix}_documents"] = []
                    st.session_state[f"{prefix}_doc_ids"] = []
                    st.session_state[f"{prefix}_doc_metadata"] = []
                    clear_chunk_cache(collection_key)
                    st.success("All documents cleared")
                    st.rerun()
            with col_b:
                if st.button("Export JSON", key=f"export_{collection_key}", use_container_width=True):
                    export_data = {
                        "count": len(st.session_state.get(f"{prefix}_documents", [])),
                        "documents": [
                            {"zotero_id": st.session_state[f"{prefix}_doc_ids"][i],
                             "title": st.session_state[f"{prefix}_doc_metadata"][i].get('title')}
                            for i in range(len(st.session_state.get(f"{prefix}_documents", [])))
                        ]
                    }
                    st.download_button("Download", json.dumps(export_data, indent=2),
                                     f"{collection_key}_documents.json", "application/json")

# ==================== MAIN TABS ====================
tabs = st.tabs([col["name"] for col in COLLECTIONS])

for tab_idx, (tab, col_info) in enumerate(zip(tabs, COLLECTIONS)):
    with tab:
        collection_key = col_info["collection_key"]
        prefix = f"col_{collection_key}"
        
        st.header(f"{col_info['name']} Collection")
        
        # ==================== LOAD FROM ZOTERO BUTTON ====================
        col_load1, col_load2 = st.columns([3, 1])
        with col_load1:
            if st.button(f"Load from Zotero - {col_info['name']}", type="primary", key=f"load_btn_{collection_key}"):
                if not zotero_library_id or not zotero_api_key:
                    st.error("Zotero Library ID and API Key must be set in secrets.")
                else:
                    collection_zotero_key = get_collection_zotero_key(collection_key)
                    if not collection_zotero_key:
                        st.error(f"Zotero collection key for '{col_info['name']}' not found in secrets.")
                    else:
                        with st.spinner(f"Loading documents from {col_info['name']}... This may take a while for large collections."):
                            try:
                                zot = zotero.Zotero(zotero_library_id, zotero_library_type, zotero_api_key)
                                items = zot.everything(zot.collection_items(collection_zotero_key))
                                
                                new_count = 0
                                for item in items:
                                    item_type = item['data']['itemType']
                                    title = item['data'].get('title', 'Untitled')
                                    low_relevance = is_item_low_relevance(item)
                                    zotero_id = item['key']
                                    
                                    if article_exists(zotero_id, None, collection_key):
                                        continue
                                    
                                    if item_type in ['note', 'annotation']:
                                        continue
                                    
                                    text = f"{title}\n{item['data'].get('abstractNote', '')}"
                                    
                                    # Handle attachments (PDFs)
                                    if item_type == 'attachment' and item['data'].get('contentType') == 'application/pdf':
                                        file_url = f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}/items/{zotero_id}/file?key={zotero_api_key}"
                                        try:
                                            resp = requests.get(file_url, timeout=15)
                                            if resp.status_code == 200:
                                                pdf_text = extract_pdf_text(resp.content)
                                                text = f"{title}\n{pdf_text}"
                                        except:
                                            pass
                                    
                                    success, msg = add_article(zotero_id, text, title, item_type, 
                                                             item['data'].get('abstractNote', ''), collection_key, low_relevance)
                                    if success:
                                        new_count += 1
                                
                                # Refresh session state
                                documents, doc_ids, doc_metadata = get_all_articles(collection_key)
                                st.session_state[f"{prefix}_documents"] = documents
                                st.session_state[f"{prefix}_doc_ids"] = doc_ids
                                st.session_state[f"{prefix}_doc_metadata"] = doc_metadata
                                
                                clear_chunk_cache(collection_key)
                                ensure_chunk_index_for_collection(collection_key)
                                
                                st.success(f"Loaded {new_count} new documents. Total: {len(documents)}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error loading from Zotero: {str(e)}")
        
        st.divider()
        
        # ==================== CHAT INTERFACE ====================
        st.subheader("Ask Questions")
        
        if f"{prefix}_messages" not in st.session_state:
            st.session_state[f"{prefix}_messages"] = []
        
        for message in st.session_state[f"{prefix}_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Processing logic
        if st.session_state.get(f'{prefix}_processing') and st.session_state.get(f'{prefix}_pending_prompt'):
            pending = st.session_state[f'{prefix}_pending_prompt']
            try:
                with st.spinner("Thinking..."):
                    cache_key = f"{collection_key}:{pending.lower().strip()}"
                    if cache_key in st.session_state.query_cache:
                        result = st.session_state.query_cache[cache_key]
                    else:
                        if not ensure_chunk_index_for_collection(collection_key):
                            result = "No documents loaded yet. Please load from Zotero first."
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
                                relevant_chunks, seen_docs, st.session_state.get('use_summaries', True)
                            )

                            if not client:
                                result = "OpenAI API key not configured."
                            else:
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": f"You are a helpful assistant knowledgeable about {col_info['name']}."},
                                        {"role": "user", "content": f"Context from documents:\n{context}\n\nQuery: {pending}"}
                                    ],
                                    stream=True
                                )
                                result = "".join([chunk.choices[0].delta.content or "" for chunk in response])

                                if cited_docs:
                                    result += "\n\n---\n**Sources:**\n"
                                    for doc in cited_docs:
                                        result += f"- {doc['title']} (Relevance: {doc['similarity']:.1%})\n"
                    
                    st.session_state.query_cache[cache_key] = result
            except Exception as e:
                result = f"Error generating response: {str(e)}"

            assistant_idx = st.session_state.get(f'{prefix}_pending_assistant_index')
            if assistant_idx is not None:
                st.session_state[f"{prefix}_messages"][assistant_idx]['content'] = result
            
            st.session_state[f'{prefix}_pending_prompt'] = None
            st.session_state[f'{prefix}_processing'] = False
            st.rerun()

        # Chat input
        if prompt := st.chat_input(f"Ask anything about {col_info['name']}...", key=f"chat_input_{collection_key}"):
            st.session_state[f"{prefix}_messages"].append({"role": "user", "content": prompt})
            st.session_state[f"{prefix}_messages"].append({"role": "assistant", "content": "Thinking..."})
            st.session_state[f'{prefix}_pending_prompt'] = prompt
            st.session_state[f'{prefix}_pending_assistant_index'] = len(st.session_state[f"{prefix}_messages"]) - 1
            st.session_state[f'{prefix}_processing'] = True
            st.rerun()

with bottom():
    st.caption("Zotero Library Source: https://www.zotero.org/groups/6420515/abpdu_workflow_automation-article_query_tool/collections/LRILZKMS/collection")