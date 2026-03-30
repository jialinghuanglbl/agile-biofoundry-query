import streamlit as st
from pyzotero import zotero
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import PyPDF2
import requests
import io
import os
import json
from streamlit_extras.bottom_container import bottom
from article_storage import (
    load_articles,
    save_articles,
    article_exists,
    add_article,
    get_all_articles,
    remove_article,
    clear_all_articles,
    rename_article
)
from document_retrieval import (
    create_chunked_documents,
    retrieve_relevant_chunks,
    format_context_from_chunks
)

# ==================== COLLECTION CONFIGURATION ====================
# Define your Zotero collections here
COLLECTIONS = [
    {"name": "Agile BioFoundry", "collection_key": "agile"},
    {"name": "ABPDU", "collection_key": "abpdu"},
]

# ==================== HELPER FUNCTIONS ====================

def extract_pdf_text(pdf_content):
    """Extract text from PDF content, inserting page number markers.

    Each page is prefixed with "Page N" so that downstream chunking and
    retrieval can identify the page where a snippet came from.
    """
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
    """Safely retrieve a secret from Streamlit secrets"""
    try:
        return st.secrets[name]
    except Exception:
        return default


def get_collection_zotero_key(collection_key: str) -> str:
    """Get Zotero collection key from secrets by collection name"""
    secret_key = f"zotero_collection_{collection_key}"
    return _safe_secret(secret_key)


def is_item_low_relevance(item: dict) -> bool:
    """Check if Zotero item has low-relevance tag."""
    tags = item.get('data', {}).get('tags', []) or []
    for tag in tags:
        tag_value = tag.get('tag') if isinstance(tag, dict) else str(tag)
        normalized = tag_value.strip().lower().replace('_', ' ').replace('-', ' ')
        if normalized in ['low relevance', 'low-relevance', 'low_relevance', 'lowrelevance']:
            return True
        # Fallback: any tag containing both key terms
        if 'low' in normalized and 'relevance' in normalized:
            return True
    return False


def init_session_state_for_collection(collection_key: str) -> None:
    """Initialize session state keys for a collection"""
    prefix = f"col_{collection_key}"

    if f"{prefix}_documents" not in st.session_state:
        documents, doc_ids, doc_metadata = get_all_articles(collection_key)
        st.session_state[f"{prefix}_documents"] = documents
        st.session_state[f"{prefix}_doc_ids"] = doc_ids
        st.session_state[f"{prefix}_doc_metadata"] = doc_metadata
        
        # Fit TF-IDF if documents exist
        if documents:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            st.session_state[f"{prefix}_vectorizer"] = vectorizer
            st.session_state[f"{prefix}_tfidf_matrix"] = tfidf_matrix
            
            # Build chunks using the standard function
            ensure_chunk_index_for_collection(collection_key)
        else:
            st.session_state[f"{prefix}_doc_ids"] = []
            st.session_state[f"{prefix}_doc_metadata"] = []


def ensure_chunk_index_for_collection(collection_key: str) -> bool:
    """Ensure chunk index for a collection exists"""
    prefix = f"col_{collection_key}"
    
    if (st.session_state.get(f'{prefix}_chunk_vectorizer') is not None and 
        st.session_state.get(f'{prefix}_chunk_tfidf_matrix') is not None and 
        st.session_state.get(f'{prefix}_chunks')):
        return True

    # Need documents to build index
    if f'{prefix}_documents' not in st.session_state or not st.session_state[f'{prefix}_documents']:
        return False

    chunk_size = st.session_state.get('chunk_size', 800)
    overlap = st.session_state.get('overlap', 150)
    summary_sentences = st.session_state.get('summary_sentences', 3)
    use_summaries = st.session_state.get('use_summaries', True)

    chunks, doc_id_mapping = create_chunked_documents(
        st.session_state[f'{prefix}_documents'],
        st.session_state[f'{prefix}_doc_ids'],
        st.session_state[f'{prefix}_doc_metadata'],
        chunk_size=chunk_size,
        overlap=overlap,
        summary_sentences=summary_sentences
    )

    if not chunks:
        st.session_state[f"{prefix}_chunks"] = []
        st.session_state[f"{prefix}_doc_id_mapping"] = []
        return False

    st.session_state[f"{prefix}_chunks"] = chunks
    st.session_state[f"{prefix}_doc_id_mapping"] = doc_id_mapping

    chunk_texts = [c['summary'] if use_summaries and c.get('summary') else c['text'] for c in chunks]
    chunk_vectorizer = TfidfVectorizer(stop_words='english')
    chunk_tfidf_matrix = chunk_vectorizer.fit_transform(chunk_texts)
    st.session_state[f"{prefix}_chunk_vectorizer"] = chunk_vectorizer
    st.session_state[f"{prefix}_chunk_tfidf_matrix"] = chunk_tfidf_matrix

    st.success(f"Auto-rebuilt index with {len(chunks)} chunks")
    return True


# ==================== MAIN APP ====================

st.set_page_config(page_title="Agile Biofoundry & ABPDU Query Tool", layout="wide")
st.title("Agile Biofoundry & ABPDU Query Tool")

# Load credentials from secrets
zotero_library_id = _safe_secret("zotero_library_id")
zotero_api_key = _safe_secret("zotero_api_key")
zotero_library_type = _safe_secret("zotero_library_type", "user")
openai_api_key = _safe_secret("openai_api_key")

# Initialize OpenAI client if key is present
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Initialize global retrieval settings (same for all collections)
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 600
if 'overlap' not in st.session_state:
    st.session_state.overlap = 100
if 'summary_sentences' not in st.session_state:
    st.session_state.summary_sentences = 2
if 'use_summaries' not in st.session_state:
    st.session_state.use_summaries = True
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}

# Initialize session state for each collection
for col_info in COLLECTIONS:
    init_session_state_for_collection(col_info['collection_key'])

# ==================== SIDEBAR: DOCUMENT MANAGEMENT ====================
with st.sidebar:
    
    
    st.header("Collections & Documents")
    
    for col_info in COLLECTIONS:
        collection_key = col_info["collection_key"]
        collection_name = col_info["name"]
        prefix = f"col_{collection_key}"
        
        with st.expander(f"{collection_name}", expanded=False):
            # Collection metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(st.session_state[f"{prefix}_documents"]))
            
            if st.session_state[f"{prefix}_documents"]:
                # Search documents
                search_term = st.text_input(f"Search {collection_name}", key=f"sidebar_search_{collection_key}", label_visibility="collapsed", placeholder="Search...")
                
                # Filter documents
                filtered_indices = []
                for idx, metadata in enumerate(st.session_state[f"{prefix}_doc_metadata"]):
                    title = metadata.get('title', 'Untitled')
                    if search_term.lower() in title.lower():
                        filtered_indices.append(idx)
                
                if not search_term:
                    filtered_indices = list(range(len(st.session_state[f"{prefix}_doc_metadata"])))
                
                st.caption(f"Showing {len(filtered_indices)} of {len(st.session_state[f'{prefix}_documents'])} docs")
                
                # Display document list with quick actions
                for idx in filtered_indices:
                    metadata = st.session_state[f"{prefix}_doc_metadata"][idx]
                    title = metadata.get('title', 'Untitled')[:40]
                    
                    with st.popover(f"{title}", use_container_width=True):
                        full_title = st.session_state[f"{prefix}_doc_metadata"][idx].get('title', 'Untitled')
                        item_type = st.session_state[f"{prefix}_doc_metadata"][idx].get('itemType', 'Unknown')
                        zotero_id = st.session_state[f"{prefix}_doc_ids"][idx]
                        
                        st.write(f"**{full_title}**")
                        st.caption(f"Type: {item_type}")
                        st.caption(f"ID: {zotero_id}")
                        
                        # Rename
                        st.subheader("Rename", divider=True)
                        new_title = st.text_input("New title", value=full_title, key=f"sidebar_rename_input_{collection_key}_{idx}", label_visibility="collapsed")
                        if st.button("Rename", key=f"sidebar_rename_{collection_key}_{idx}", use_container_width=True):
                            success, msg = rename_article(zotero_id, new_title, collection_key)
                            if success:
                                st.session_state[f"{prefix}_doc_metadata"][idx]['title'] = new_title
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                        
                        # Delete
                        st.subheader("Delete", divider=True)
                        if st.button("Delete", key=f"sidebar_delete_{collection_key}_{idx}", use_container_width=True):
                            remove_article(zotero_id, collection_key)
                            
                            del st.session_state[f"{prefix}_documents"][idx]
                            del st.session_state[f"{prefix}_doc_ids"][idx]
                            del st.session_state[f"{prefix}_doc_metadata"][idx]
                            
                            # Clear existing index to force rebuild
                            for key in [f'{prefix}_chunks', f'{prefix}_doc_id_mapping', f'{prefix}_chunk_vectorizer', f'{prefix}_chunk_tfidf_matrix']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            # Rebuild index using the standard function
                            ensure_chunk_index_for_collection(collection_key)
                            
                            st.success("Document deleted and index rebuilt")
                            st.rerun()
                
                # Bulk actions
                st.divider()
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("Clear All", key=f"sidebar_clear_all_{collection_key}", use_container_width=True):
                        clear_all_articles(collection_key)
                        st.session_state[f"{prefix}_documents"] = []
                        st.session_state[f"{prefix}_doc_ids"] = []
                        st.session_state[f"{prefix}_doc_metadata"] = []
                        for key in [f'{prefix}_vectorizer', f'{prefix}_tfidf_matrix', f'{prefix}_chunks', f'{prefix}_doc_id_mapping', f'{prefix}_chunk_vectorizer', f'{prefix}_chunk_tfidf_matrix']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.success("All cleared")
                        st.rerun()
                
                with col_b:
                    if st.button("Export", key=f"sidebar_export_{collection_key}", use_container_width=True):
                        export_data = {
                            "count": len(st.session_state[f"{prefix}_documents"]),
                            "documents": [
                                {
                                    "zotero_id": st.session_state[f"{prefix}_doc_ids"][i],
                                    "title": st.session_state[f"{prefix}_doc_metadata"][i].get('title', 'Untitled'),
                                    "type": st.session_state[f"{prefix}_doc_metadata"][i].get('itemType', 'Unknown')
                                }
                                for i in range(len(st.session_state[f"{prefix}_documents"]))
                            ]
                        }
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"zotero_documents_{collection_key}.json",
                            mime="application/json",
                            key=f"sidebar_download_{collection_key}",
                            use_container_width=True
                        )
                
                # Cache management
                st.divider()
                cache_col1, cache_col2 = st.columns(2)
                with cache_col1:
                    cache_size = len([k for k in st.session_state.query_cache.keys() if k.startswith(f"{collection_key}:")])
                    st.metric("Cached Queries", cache_size)
                with cache_col2:
                    if st.button("Clear Query Cache", key=f"clear_cache_{collection_key}", use_container_width=True):
                        # Remove cache entries for this collection
                        keys_to_remove = [k for k in st.session_state.query_cache.keys() if k.startswith(f"{collection_key}:")]
                        for key in keys_to_remove:
                            del st.session_state.query_cache[key]
                        st.success(f"Cleared {len(keys_to_remove)} cached queries")
                        st.rerun()
            else:
                st.caption("No documents yet. Load from Zotero tab.")
# ==================== MAIN CONTENT WITH TABS ====================

tabs = st.tabs([col["name"] for col in COLLECTIONS])

for tab, col_info in zip(tabs, COLLECTIONS):
    with tab:
        collection_key = col_info["collection_key"]
        collection_name = col_info["name"]
        prefix = f"col_{collection_key}"
        
        st.header(f"{collection_name} Collection")
        
        
        
        # ==================== LOAD FROM ZOTERO ====================
        st.divider()
        
        if st.button(f"Load from Zotero - {collection_name}", type="primary", key=f"load_{collection_key}"):
            if not zotero_library_id or not zotero_api_key:
                st.error("Zotero Library ID and API Key must be set in Streamlit secrets.")
            else:
                collection_zotero_key = get_collection_zotero_key(collection_key)
                if not collection_zotero_key:
                    st.error(f"Zotero collection key for '{collection_name}' not found in secrets (zotero_collection_{collection_key}).")
                else:
                    with st.spinner(f"Loading documents from {collection_name}..."):
                        try:
                            zot = zotero.Zotero(zotero_library_id, zotero_library_type, zotero_api_key)
                            # Ensure we fetch all items (pyzotero paginates by default)
                            items = zot.everything(zot.collection_items(collection_zotero_key))
                            
                            documents = []
                            doc_ids = []
                            doc_metadata = []
                            duplicates = []
                            new_count = 0
                            
                            skipped_items = {
                                'wrong_type': [],
                                'duplicate': [],
                                'no_content': [],
                                'errors': [],
                                'temp_errors': []  # For temporary issues like 502
                            }

                            progress_bar = st.progress(0)
                            total_raw_items = len(items)
                            total_processed = 0

                            for item_idx, item in enumerate(items):
                                # Guard against division by zero
                                if total_raw_items:
                                    progress_bar.progress(min(1.0, (item_idx + 1) / total_raw_items))
                                else:
                                    progress_bar.progress(1.0)
                                
                                item_type = item['data']['itemType']
                                title = item['data'].get('title', 'Untitled')
                                low_relevance = is_item_low_relevance(item)
                                
                                # Skip notes and annotations (don't count as processed)
                                if item_type in ['note', 'annotation']:
                                    continue
                                # Count items that we actually consider for import
                                total_processed += 1
                                
                                # Check duplicates
                                if article_exists(item['key'], None, collection_key):
                                    duplicates.append(title)
                                    skipped_items['duplicate'].append(title)
                                    continue

                                try:
                                    # Handle standalone attachments
                                    if item_type == 'attachment':
                                        link_mode = item['data'].get('linkMode', '')
                                        content_type = item['data'].get('contentType', '')
                                        
                                        if 'application/pdf' in content_type or link_mode in ['linked_file', 'imported_file', 'imported_url']:
                                            file_url = f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}/items/{item['key']}/file?key={zotero_api_key}"
                                            try:
                                                response = requests.get(file_url, timeout=10)
                                                if response.status_code == 200:
                                                    actual_content_type = response.headers.get('Content-Type', '')
                                                    if 'application/pdf' in actual_content_type:
                                                        pdf_text = extract_pdf_text(response.content)
                                                        if not pdf_text.startswith("Error") and pdf_text.strip():
                                                            text = f"{title}\n{pdf_text}"
                                                            success, msg = add_article(item['key'], text, title, 'attachment', '', collection_key, low_relevance=low_relevance)
                                                            if success:
                                                                documents.append(text)
                                                                doc_ids.append(item['key'])
                                                                doc_metadata.append({
                                                                    'title': title,
                                                                    'itemType': 'attachment (PDF)',
                                                                    'abstract': '',
                                                                    'low_relevance': low_relevance
                                                                })
                                                                new_count += 1
                                                        else:
                                                            skipped_items['no_content'].append(f"{title} (PDF extraction failed)")
                                                    else:
                                                        skipped_items['wrong_type'].append(f"{title} (non-PDF attachment)")
                                            except Exception as e:
                                                skipped_items['errors'].append(f"{title}: {str(e)}")
                                        else:
                                            skipped_items['wrong_type'].append(f"{title} (non-PDF attachment)")
                                        continue
                                    
                                    # Track unsupported types
                                    # Include video/audio recordings so transcripts in child notes are processed
                                    if item_type not in [
                                        'journalArticle', 'webpage', 'report', 'conferencePaper',
                                        'book', 'bookSection', 'preprint', 'document', 'presentation',
                                        'videoRecording', 'audioRecording', 'blogPost', 'non-PDF attachment'
                                    ]:
                                        skipped_items['wrong_type'].append(f"{title} ({item_type})")
                                        continue

                                    # Extract metadata
                                    abstract = item['data'].get('abstractNote', '')
                                    notes = []

                                    # Fetch all children (notes, attachments) for this item
                                    children = []
                                    max_retries = 3
                                    for attempt in range(max_retries):
                                        try:
                                            children = zot.everything(zot.children(item['key']))
                                            break  # Success, exit retry loop
                                        except Exception as child_err:
                                            if attempt < max_retries - 1:  # Not the last attempt
                                                import time
                                                time.sleep(1 * (attempt + 1))  # Exponential backoff: 1s, 2s, 3s
                                            else:
                                                # All retries failed, continue without children
                                                children = []
                                                if "502" in str(child_err) or "Bad Gateway" in str(child_err):
                                                    # Temporary server error, log separately
                                                    skipped_items['temp_errors'].append(f"{title}: Could not fetch notes/attachments (temporary server error)")
                                                else:
                                                    # Other persistent errors
                                                    skipped_items['temp_errors'].append(f"{title}: Could not fetch notes/attachments ({str(child_err)})")
                                    
                                    for child in children:
                                        if child['data']['itemType'] == 'note':
                                            notes.append(child['data'].get('note', ''))

                                    text = f"{title}\n{abstract}\n{' '.join(notes)}"

                                    # Process attachments
                                    for child in children:
                                        if child['data']['itemType'] == 'attachment':
                                            link_mode = child['data'].get('linkMode', '')
                                            if link_mode in ['linked_file', 'imported_file', 'imported_url']:
                                                file_url = f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}/items/{child['key']}/file?key={zotero_api_key}"
                                                attachment_success = False
                                                for attempt in range(3):  # Retry attachment downloads
                                                    try:
                                                        response = requests.get(file_url, timeout=10)
                                                        if response.status_code == 200:
                                                            content_type = response.headers.get('Content-Type', '')
                                                            if 'application/pdf' in content_type:
                                                                pdf_text = extract_pdf_text(response.content)
                                                                if not pdf_text.startswith("Error"):
                                                                    text += f"\n{pdf_text}"
                                                                    attachment_success = True
                                                            elif 'text/html' in content_type:
                                                                text += f"\n{response.text}"
                                                                attachment_success = True
                                                        break  # Success or final attempt
                                                    except Exception as attach_err:
                                                        if attempt < 2:  # Not the last attempt
                                                            import time
                                                            time.sleep(1 * (attempt + 1))
                                                        # Continue to next attempt or give up
                                                
                                                if not attachment_success and response.status_code != 200:
                                                    # Log attachment fetch failure but don't fail the whole item
                                                    pass

                                    if text.strip():
                                        success, msg = add_article(item['key'], text, title, item_type, abstract, collection_key, low_relevance=low_relevance)
                                        if success:
                                            documents.append(text)
                                            doc_ids.append(item['key'])
                                            doc_metadata.append({
                                                'title': title,
                                                'itemType': item_type,
                                                'abstract': abstract[:200] if abstract else '',
                                                'low_relevance': low_relevance
                                            })
                                            new_count += 1
                                    else:
                                        skipped_items['no_content'].append(title)
                                        
                                except Exception as e:
                                    skipped_items['errors'].append(f"{title}: {str(e)}")

                            progress_bar.empty()

                            # Store report and reload
                            st.session_state[f"{prefix}_last_load_report"] = {
                                'new_count': new_count,
                                'duplicates': len(duplicates),
                                'skipped_items': skipped_items,
                                'total_processed': total_processed,
                                'total_raw_items': total_raw_items
                            }

                            if not documents and not duplicates:
                                st.error("No documents found in the Zotero collection.")
                            else:
                                # Reload all articles from storage
                                all_documents, all_doc_ids, all_doc_metadata = get_all_articles(collection_key)
                                
                                st.session_state[f"{prefix}_documents"] = all_documents
                                st.session_state[f"{prefix}_doc_ids"] = all_doc_ids
                                st.session_state[f"{prefix}_doc_metadata"] = all_doc_metadata

                                # Rebuild index
                                if all_documents:
                                    vectorizer = TfidfVectorizer(stop_words='english')
                                    tfidf_matrix = vectorizer.fit_transform(all_documents)
                                    st.session_state[f"{prefix}_vectorizer"] = vectorizer
                                    st.session_state[f"{prefix}_tfidf_matrix"] = tfidf_matrix
                                    
                                    # Clear existing chunk index and rebuild
                                    for key in [f'{prefix}_chunks', f'{prefix}_doc_id_mapping', f'{prefix}_chunk_vectorizer', f'{prefix}_chunk_tfidf_matrix']:
                                        if key in st.session_state:
                                            del st.session_state[key]
                                    
                                    ensure_chunk_index_for_collection(collection_key)
                                    
                                    # Get chunk count for success message
                                    chunk_count = len(st.session_state.get(f"{prefix}_chunks", []))
                                    st.success(f"Auto-rebuilt index with {chunk_count} chunks")

                                message = f"Loaded {new_count} new documents from {collection_name} (Total: {len(all_documents)} documents)"
                                if duplicates:
                                    message += f". Skipped {len(duplicates)} duplicate(s)."
                                st.success(message)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading Zotero collection: {str(e)}")
        
        # Display load report
        if f"{prefix}_last_load_report" in st.session_state:
            report = st.session_state[f"{prefix}_last_load_report"]
            
            st.success(f"Last load: {report['new_count']} new documents added from {report['total_processed']} processed items ({report.get('total_raw_items', report['total_processed'])} total in Zotero)")
            
            if report['duplicates'] > 0:
                st.info(f"Skipped {report['duplicates']} duplicates (already in database)")
            
            skipped = report['skipped_items']
            
            if skipped['wrong_type']:
                st.warning(f"Skipped {len(skipped['wrong_type'])} items due to item type:")
                with st.expander("View skipped item types"):
                    for item in skipped['wrong_type']:
                        st.write(f"- {item}")

            if skipped['no_content']:
                st.warning(f"Skipped {len(skipped['no_content'])} items with no extractable content:")
                with st.expander("View items with no content"):
                    for item in skipped['no_content']:
                        st.write(f"- {item}")

            if skipped['temp_errors']:
                st.warning(f"Encountered {len(skipped['temp_errors'])} temporary issues (items loaded without notes/attachments):")
                with st.expander("View temporary issues"):
                    for item in skipped['temp_errors']:
                        st.write(f"- {item}")
            
            if skipped['errors']:
                # Check if there are 502 errors
                has_502_errors = any("502" in error or "Bad Gateway" in error for error in skipped['errors'])
                
                if has_502_errors:
                    st.error(f"Encountered {len(skipped['errors'])} permanent errors (including server issues):")
                    st.info("💡 **Tip:** Some errors may be temporary. Try loading again in a few minutes.")
                    col_retry, col_clear = st.columns([1, 1])
                    with col_retry:
                        if st.button("Retry Load", key=f"retry_load_{collection_key}", type="primary"):
                            # Clear the report and trigger a reload
                            del st.session_state[f"{prefix}_last_load_report"]
                            st.rerun()
                    with col_clear:
                        pass  # Space for future buttons
                else:
                    st.error(f"Encountered {len(skipped['errors'])} permanent errors:")
                
                with st.expander("View permanent errors"):
                    for item in skipped['errors']:
                        st.write(f"- {item}")
            
            if st.button("Clear Load Report", key=f"clear_report_{collection_key}"):
                del st.session_state[f"{prefix}_last_load_report"]
                st.rerun()
        
        # ==================== CHAT INTERFACE ====================
        st.divider()
        st.header(f"Query {collection_name}")
        
        # Initialize chat history for this collection
        if f"{prefix}_messages" not in st.session_state:
            st.session_state[f"{prefix}_messages"] = []
        
        # Display chat
        for message in st.session_state[f"{prefix}_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Process pending prompt
        if st.session_state.get(f'{prefix}_processing') and st.session_state.get(f'{prefix}_pending_prompt'):
            pending = st.session_state[f'{prefix}_pending_prompt']
            try:
                with st.spinner("Typing..."):
                    # Check cache first
                    cache_key = f"{collection_key}:{pending.lower().strip()}"
                    if cache_key in st.session_state.query_cache:
                        result = st.session_state.query_cache[cache_key]
                    else:
                        if not ensure_chunk_index_for_collection(collection_key):
                            result = "No indexed documents available to answer the query. Please load the Zotero library."
                        else:
                            k_chunks = st.session_state.get('k_chunks', 3)
                            similarity_threshold = 0.12

                            relevant_chunks, seen_docs = retrieve_relevant_chunks(
                                pending,
                                st.session_state[f"{prefix}_chunk_vectorizer"],
                                st.session_state[f"{prefix}_chunk_tfidf_matrix"],
                                st.session_state[f"{prefix}_chunks"],
                                st.session_state[f"{prefix}_doc_id_mapping"],
                                k=k_chunks,
                                similarity_threshold=similarity_threshold
                            )

                            use_summaries = st.session_state.get('use_summaries', True)
                            context, cited_docs = format_context_from_chunks(relevant_chunks, seen_docs, use_summaries=use_summaries)

                            if not context or context == "":
                                context = "No relevant documents found."

                            if client is None:
                                result = "OpenAI API key is not set. Please add it to Streamlit secrets to enable the assistant."
                            else:
                                result = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": f"You are a helpful assistant knowledgeable about {collection_name}. Use the provided context to answer the query comprehensively."},
                                        {"role": "user", "content": f"Context from knowledge base:\n{context}\n\nQuery: {pending}"}
                                    ]
                                ).choices[0].message.content

                                if cited_docs:
                                    result += "\n\n---\n**Sources:**\n"
                                    for doc in cited_docs:
                                        extra = []
                                        if doc.get('pages'):
                                            extra.append("pages " + ",".join(doc['pages']))
                                        if doc.get('timestamps'):
                                            extra.append("times " + ",".join(doc['timestamps']))
                                        extras = f" ({'; '.join(extra)})" if extra else ""
                                        result += f"- {doc['title']}{extras} (ID: {doc['id']}, Relevance: {doc['similarity']:.2%}, Chunks: {doc['chunk_count']})\n"
                        
                        # Cache the result
                        st.session_state.query_cache[cache_key] = result
            except Exception as e:
                result = f"Error generating response: {str(e)}"

            # Update message
            assistant_idx = st.session_state.get(f'{prefix}_pending_assistant_index')
            if assistant_idx is not None and assistant_idx < len(st.session_state[f"{prefix}_messages"]):
                st.session_state[f"{prefix}_messages"][assistant_idx]['content'] = result
            else:
                st.session_state[f"{prefix}_messages"].append({"role": "assistant", "content": result})

            st.session_state[f'{prefix}_pending_prompt'] = None
            st.session_state[f'{prefix}_pending_assistant_index'] = None
            st.session_state[f'{prefix}_processing'] = False

            st.rerun()
        
        # User input
        if prompt := st.chat_input(f"Ask about {collection_name}:", key=f"chat_input_{collection_key}"):
            st.session_state[f"{prefix}_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state[f"{prefix}_messages"].append({"role": "assistant", "content": "Bot is thinking..."})
            st.session_state[f'{prefix}_pending_prompt'] = prompt
            st.session_state[f'{prefix}_pending_assistant_index'] = len(st.session_state[f"{prefix}_messages"]) - 1
            st.session_state[f'{prefix}_processing'] = True
            st.rerun()

with bottom():
    st.write("Zotero Library Source: https://www.zotero.org/groups/6420515/abpdu_workflow_automation-article_query_tool/collections/LRILZKMS/collection") 