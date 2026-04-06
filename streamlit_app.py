import streamlit as st
from pyzotero import zotero
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from groq import Groq
import PyPDF2
import requests
import io
import json
import hashlib
from streamlit_extras.bottom_container import bottom

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

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

SIDEBAR_PAGE_SIZE = 10   # documents rendered per page in sidebar
MAX_CONTEXT_CHARS = 800  # max chars per chunk sent to Groq


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


# ==================== INDEX MANAGEMENT ====================
def _build_index_for_collection(collection_key: str) -> bool:
    prefix = f"col_{collection_key}"

    if f"{prefix}_documents" not in st.session_state:
        documents, doc_ids, doc_metadata = get_all_articles(collection_key)
        st.session_state[f"{prefix}_documents"] = documents
        st.session_state[f"{prefix}_doc_ids"] = doc_ids
        st.session_state[f"{prefix}_doc_metadata"] = doc_metadata

    loaded = load_chunk_index(collection_key)
    if loaded:
        chunks, doc_id_mapping, vectorizer, faiss_index = loaded
        st.session_state[f"{prefix}_chunks"] = chunks
        st.session_state[f"{prefix}_doc_id_mapping"] = doc_id_mapping
        st.session_state[f"{prefix}_chunk_vectorizer"] = vectorizer
        st.session_state[f"{prefix}_faiss_index"] = faiss_index
        return True

    documents = st.session_state.get(f"{prefix}_documents", [])
    if not documents:
        return False

    chunks, doc_id_mapping = create_chunked_documents(
        documents,
        st.session_state[f"{prefix}_doc_ids"],
        st.session_state[f"{prefix}_doc_metadata"],
        chunk_size=400, overlap=80, summary_sentences=1
    )
    if not chunks:
        return False

    chunk_texts = [c.get('summary', c['text']) for c in chunks]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    if FAISS_AVAILABLE:
        chunk_vectors = tfidf_matrix.toarray().astype('float32')
        faiss.normalize_L2(chunk_vectors)
        faiss_index = faiss.IndexFlatIP(chunk_vectors.shape[1])
        faiss_index.add(chunk_vectors)
    else:
        faiss_index = None

    st.session_state[f"{prefix}_chunks"] = chunks
    st.session_state[f"{prefix}_doc_id_mapping"] = doc_id_mapping
    st.session_state[f"{prefix}_chunk_vectorizer"] = vectorizer
    st.session_state[f"{prefix}_faiss_index"] = faiss_index

    save_chunk_index(chunks, doc_id_mapping, vectorizer, faiss_index, collection_key)
    return True


def _bootstrap_indexes():
    if st.session_state.get("_indexes_bootstrapped"):
        return
    for col_info in COLLECTIONS:
        _build_index_for_collection(col_info["collection_key"])
    st.session_state["_indexes_bootstrapped"] = True


def invalidate_index(collection_key: str):
    prefix = f"col_{collection_key}"
    for key in ["_chunks", "_doc_id_mapping", "_chunk_vectorizer", "_faiss_index"]:
        st.session_state.pop(f"{prefix}{key}", None)
    clear_chunk_cache(collection_key)
    st.session_state.pop("_indexes_bootstrapped", None)


def index_ready(collection_key: str) -> bool:
    prefix = f"col_{collection_key}"
    return (
        f"{prefix}_chunks" in st.session_state
        and f"{prefix}_chunk_vectorizer" in st.session_state
        and st.session_state.get(f"{prefix}_faiss_index") is not None
    )


# ==================== CONTEXT BUILDER ====================
def build_capped_context(relevant_chunks, seen_docs, use_summaries: bool) -> tuple:
    """
    Same as format_context_from_chunks but caps each chunk's text at
    MAX_CONTEXT_CHARS before sending to the model. The full text is still
    stored in session state for display purposes.
    """
    context_parts = []
    chunks_by_doc = {}
    for chunk in relevant_chunks:
        chunks_by_doc.setdefault(chunk['doc_id'], []).append(chunk)

    for doc_id, chunks in chunks_by_doc.items():
        title = chunks[0]['doc_title']
        context_parts.append(f"\n**Source: {title}**")
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('summary') if use_summaries and chunk.get('summary') else chunk['text']
            # Cap here — reduces tokens sent to Groq significantly
            text = text[:MAX_CONTEXT_CHARS]
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
            'similarity': info['max_similarity'],
            'chunk_count': info['chunk_count']
        }
        if info.get('pages'):
            entry['pages'] = sorted(info['pages'], key=int)
        if info.get('timestamps'):
            entry['timestamps'] = sorted(info['timestamps'])
        cited_docs.append(entry)

    return context, cited_docs


# ==================== APP SETUP ====================
st.set_page_config(page_title="Agile Biofoundry & ABPDU Query Tool", layout="wide")

_bootstrap_indexes()

st.title("Agile Biofoundry & ABPDU Query Tool")

zotero_library_id = _safe_secret("zotero_library_id")
zotero_api_key    = _safe_secret("zotero_api_key")
zotero_library_type = _safe_secret("zotero_library_type", "user")
groq_api_key      = _safe_secret("groq_api_key")

client = Groq(api_key=groq_api_key) if groq_api_key else None

if 'use_summaries' not in st.session_state:
    st.session_state.use_summaries = True
if 'k_chunks' not in st.session_state:
    st.session_state.k_chunks = 3
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Collections & Documents")

    for col_info in COLLECTIONS:
        collection_key = col_info["collection_key"]
        collection_name = col_info["name"]
        prefix = f"col_{collection_key}"

        with st.expander(f"{collection_name}", expanded=False):
            docs      = st.session_state.get(f"{prefix}_doc_metadata", [])
            doc_ids   = st.session_state.get(f"{prefix}_doc_ids", [])
            doc_count = len(docs)
            st.metric("Documents", doc_count)

            # Search filter — stored in session state so it survives reruns
            # without resetting pagination
            search_key = f"search_{collection_key}"
            page_key   = f"sidebar_page_{collection_key}"
            if search_key not in st.session_state:
                st.session_state[search_key] = ""
            if page_key not in st.session_state:
                st.session_state[page_key] = 0

            search_term = st.text_input(
                f"Search {collection_name}",
                key=search_key,
                placeholder="Search documents...",
                on_change=lambda: st.session_state.update({page_key: 0})
            )

            # Build filtered index list once — no popover rendering yet
            filtered_indices = [
                i for i, m in enumerate(docs)
                if not search_term or search_term.lower() in m.get('title', '').lower()
            ]
            total_filtered = len(filtered_indices)

            # Pagination controls
            total_pages = max(1, (total_filtered + SIDEBAR_PAGE_SIZE - 1) // SIDEBAR_PAGE_SIZE)
            current_page = min(st.session_state[page_key], total_pages - 1)
            st.session_state[page_key] = current_page

            page_start = current_page * SIDEBAR_PAGE_SIZE
            page_end   = min(page_start + SIDEBAR_PAGE_SIZE, total_filtered)
            page_slice = filtered_indices[page_start:page_end]

            if total_filtered > SIDEBAR_PAGE_SIZE:
                st.caption(f"Showing {page_start + 1}–{page_end} of {total_filtered}")
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                with nav_col1:
                    if st.button("Prev", key=f"prev_{collection_key}",
                                 disabled=current_page == 0, use_container_width=True):
                        st.session_state[page_key] -= 1
                        st.rerun()
                with nav_col2:
                    st.caption(f"Page {current_page + 1} / {total_pages}")
                with nav_col3:
                    if st.button("Next", key=f"next_{collection_key}",
                                 disabled=current_page >= total_pages - 1, use_container_width=True):
                        st.session_state[page_key] += 1
                        st.rerun()

            # Render ONLY the current page slice — typically 10 popovers max
            for idx in page_slice:
                meta       = docs[idx]
                zotero_id  = doc_ids[idx]
                title_short = meta.get('title', 'Untitled')[:50]

                with st.popover(title_short, use_container_width=True):
                    full_title = meta.get('title', 'Untitled')
                    st.write(f"**{full_title}**")
                    st.caption(f"ID: {zotero_id}")

                    new_title = st.text_input(
                        "New title", value=full_title,
                        key=f"rename_in_{collection_key}_{idx}"
                    )
                    if st.button("Rename", key=f"rename_btn_{collection_key}_{idx}"):
                        success, msg = rename_article(zotero_id, new_title, collection_key)
                        if success:
                            st.session_state[f"{prefix}_doc_metadata"][idx]['title'] = new_title
                            invalidate_index(collection_key)
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                    if st.button("Delete", key=f"delete_btn_{collection_key}_{idx}", type="secondary"):
                        remove_article(zotero_id, collection_key)
                        del st.session_state[f"{prefix}_documents"][idx]
                        del st.session_state[f"{prefix}_doc_ids"][idx]
                        del st.session_state[f"{prefix}_doc_metadata"][idx]
                        invalidate_index(collection_key)
                        st.success("Document deleted")
                        st.rerun()

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Clear All", key=f"clear_all_{collection_key}", use_container_width=True):
                    clear_all_articles(collection_key)
                    st.session_state[f"{prefix}_documents"] = []
                    st.session_state[f"{prefix}_doc_ids"]   = []
                    st.session_state[f"{prefix}_doc_metadata"] = []
                    st.session_state[page_key] = 0
                    invalidate_index(collection_key)
                    st.success("All documents cleared")
                    st.rerun()
            with col_b:
                if st.button("Export JSON", key=f"export_{collection_key}", use_container_width=True):
                    export_data = {
                        "count": doc_count,
                        "documents": [
                            {
                                "zotero_id": doc_ids[i],
                                "title": docs[i].get('title')
                            }
                            for i in range(doc_count)
                        ]
                    }
                    st.download_button(
                        "Download", json.dumps(export_data, indent=2),
                        f"{collection_key}_documents.json", "application/json"
                    )


# ==================== MAIN TABS ====================
tabs = st.tabs([col["name"] for col in COLLECTIONS])

for tab_idx, (tab, col_info) in enumerate(zip(tabs, COLLECTIONS)):
    with tab:
        collection_key = col_info["collection_key"]
        prefix = f"col_{collection_key}"

        st.header(f"{col_info['name']} Collection")

        # ==================== LOAD FROM ZOTERO ====================
        if st.button(f"Load from Zotero - {col_info['name']}", type="primary", key=f"load_btn_{collection_key}"):
            if not zotero_library_id or not zotero_api_key:
                st.error("Zotero Library ID and API Key must be set in secrets.")
            else:
                collection_zotero_key = get_collection_zotero_key(collection_key)
                if not collection_zotero_key:
                    st.error(f"Zotero collection key for '{col_info['name']}' not found in secrets.")
                else:
                    with st.spinner(f"Loading documents from {col_info['name']}..."):
                        try:
                            zot = zotero.Zotero(zotero_library_id, zotero_library_type, zotero_api_key)
                            items = zot.everything(zot.collection_items(collection_zotero_key))

                            new_count = 0
                            for item in items:
                                item_type    = item['data']['itemType']
                                title        = item['data'].get('title', 'Untitled')
                                low_relevance = is_item_low_relevance(item)
                                zotero_id    = item['key']

                                if article_exists(zotero_id, None, collection_key):
                                    continue
                                if item_type in ['note', 'annotation']:
                                    continue

                                text = f"{title}\n{item['data'].get('abstractNote', '')}"

                                if item_type == 'attachment' and item['data'].get('contentType') == 'application/pdf':
                                    file_url = (
                                        f"https://api.zotero.org/{zotero_library_type}s/"
                                        f"{zotero_library_id}/items/{zotero_id}/file?key={zotero_api_key}"
                                    )
                                    try:
                                        resp = requests.get(file_url, timeout=15)
                                        if resp.status_code == 200:
                                            text = f"{title}\n{extract_pdf_text(resp.content)}"
                                    except Exception:
                                        pass

                                success, _ = add_article(
                                    zotero_id, text, title, item_type,
                                    item['data'].get('abstractNote', ''),
                                    collection_key, low_relevance
                                )
                                if success:
                                    new_count += 1

                            documents, doc_ids, doc_metadata = get_all_articles(collection_key)
                            st.session_state[f"{prefix}_documents"]    = documents
                            st.session_state[f"{prefix}_doc_ids"]      = doc_ids
                            st.session_state[f"{prefix}_doc_metadata"] = doc_metadata

                            invalidate_index(collection_key)

                            with st.spinner("Building search index..."):
                                _build_index_for_collection(collection_key)
                            st.session_state["_indexes_bootstrapped"] = True

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

        if prompt := st.chat_input(
            f"Ask anything about {col_info['name']}...",
            key=f"chat_input_{collection_key}"
        ):
            st.session_state[f"{prefix}_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            cache_key = f"{collection_key}:{prompt.lower().strip()}"

            with st.chat_message("assistant"):
                history = st.session_state[f"{prefix}_messages"]
                history_text = "\n".join(f"{msg['role']}:{msg['content']}" for msg in history)
                history_hash = hashlib.sha256(history_text.encode('utf-8')).hexdigest()
                cache_key = f"{collection_key}:{history_hash}"

                if cache_key in st.session_state.query_cache:
                    result = st.session_state.query_cache[cache_key]
                    st.markdown(result)

                elif not index_ready(collection_key):
                    result = "No documents loaded yet. Please load from Zotero first."
                    st.markdown(result)

                elif not client:
                    result = "Groq API key not configured. Add groq_api_key to your secrets."
                    st.markdown(result)

                else:
                    # FAISS search — milliseconds
                    relevant_chunks, seen_docs = retrieve_relevant_chunks(
                        prompt,
                        st.session_state[f"{prefix}_chunk_vectorizer"],
                        st.session_state[f"{prefix}_chunks"],
                        st.session_state[f"{prefix}_doc_id_mapping"],
                        faiss_index=st.session_state[f"{prefix}_faiss_index"],
                        k=st.session_state.get('k_chunks', 3),
                        similarity_threshold=0.12
                    )

                    # Cap context to MAX_CONTEXT_CHARS per chunk before sending
                    context, cited_docs = build_capped_context(
                        relevant_chunks, seen_docs,
                        st.session_state.get('use_summaries', True)
                    )

                    previous_history = history[:-1]
                    latest_prompt = history[-1]['content'] if history else prompt

                    conversation = [
                        {
                            "role": "system",
                            "content": (
                                f"You are a helpful assistant for {col_info['name']}. "
                                "Answer in 3-6 sentences unless the question genuinely requires more. "
                                "Base your answer only on the provided context. "
                                "If the context does not contain enough information, say so briefly."
                            )
                        }
                    ]
                    conversation.extend(previous_history)
                    conversation.append({
                        "role": "user",
                        "content": f"Context from documents:\n{context}\n\nQuestion: {latest_prompt}"
                    })

                    stream = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=conversation,
                        stream=True,
                        max_tokens=1024,
                    )

                    # Tokens appear in the UI as they arrive
                    result = st.write_stream(
                        chunk.choices[0].delta.content or "" for chunk in stream
                    )

                    if cited_docs:
                        sources = "\n\n---\n**Sources:**\n"
                        for doc in cited_docs:
                            sources += f"- {doc['title']} (Relevance: {doc['similarity']:.1%})\n"
                        st.markdown(sources)
                        result = (result or "") + sources

                st.session_state.query_cache[cache_key] = result

            st.session_state[f"{prefix}_messages"].append(
                {"role": "assistant", "content": result or ""}
            )

with bottom():
    st.caption("Zotero Library Source: https://www.zotero.org/groups/6420515/abpdu_workflow_automation-article_query_tool/collections/LRILZKMS/collection")