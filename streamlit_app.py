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
import base64
import re
 
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
try:
    from PIL import Image
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Image processing libraries not available: {e}")
    Image = None
    IMAGE_PROCESSING_AVAILABLE = False
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None
    FITZ_AVAILABLE = False
 
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
MAX_CONTEXT_CHARS = 800  # max chars per chunk sent to LLM
 
 
# ==================== HELPERS ====================
def extract_pdf_text(pdf_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for i, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            # Sentinel placed at END of each page's text so it stays
            # attached to the content in the same chunk rather than being
            # split off as an isolated token at the start of the next chunk.
            text += page_text.strip() + f" [PAGE:{i}] "
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"
 
 
def extract_pdf_page_previews(pdf_content: bytes, zoom: float = 1.5):
    """
    Render ALL PDF pages as base64 PNG images for storage at ingest time.
    Each entry is keyed by page number (string) so render_article_preview
    can do an O(1) lookup by chunk_page instead of scanning.
 
    FIX: removed max_pages cap so every page is available for preview.
         page value is now a string to match chunk_page strings from
         extract_page_number().
    """
    if not FITZ_AVAILABLE or not IMAGE_PROCESSING_AVAILABLE:
        return []
 
    try:
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        previews = []
        total = len(pdf_doc)
        for page_num in range(total):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            buf = io.BytesIO()
            Image.open(io.BytesIO(pix.tobytes("png"))).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            previews.append({
                "data": f"data:image/png;base64,{b64}",
                "page": str(page_num + 1),   # string — matches chunk_page strings
                "source": "preview",
                "classification": "page_preview",
                "caption": f"Page {page_num + 1} of {total}",
            })
        pdf_doc.close()
        return previews
    except Exception as e:
        print(f"[ERROR] extract_pdf_page_previews: {e}")
        return []
 
 
def dedupe_image_urls(image_urls):
    seen = set()
    deduped = []
    for img in image_urls:
        if isinstance(img, dict):
            data = img.get('data') or img.get('url') or img.get('src')
        else:
            data = str(img)
        if not data or data in seen:
            continue
        seen.add(data)
        deduped.append(img)
    return deduped
 
 
def _safe_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return default
 
 
def fetch_pdf_page_from_zotero(zotero_id: str, page_num: int, zoom: float = 1.5) -> str:
    """
    Fetch a specific page from a Zotero item's PDF attachment and render it
    as a base64 data URI.  Returns empty string on any failure.
 
    FIX: added allow_redirects=True to both requests.get calls — Zotero's
         API sometimes returns 302 to the actual cloud storage URL.
    """
    print(f"[DEBUG] fetch_pdf_page_from_zotero called with zotero_id={zotero_id}, page_num={page_num}")
 
    if not FITZ_AVAILABLE or not IMAGE_PROCESSING_AVAILABLE:
        print(f"[DEBUG] FITZ_AVAILABLE={FITZ_AVAILABLE}, IMAGE_PROCESSING_AVAILABLE={IMAGE_PROCESSING_AVAILABLE}")
        return ""
 
    try:
        zotero_library_id = _safe_secret("zotero_library_id")
        zotero_api_key = _safe_secret("zotero_api_key")
        zotero_library_type = _safe_secret("zotero_library_type", "user")
 
        print(f"[DEBUG] Secrets: library_id={zotero_library_id[:4]}..., has_key={bool(zotero_api_key)}, type={zotero_library_type}")
 
        if not zotero_library_id or not zotero_api_key:
            print("[DEBUG] Missing Zotero credentials")
            return ""
 
        file_url = (
            f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}/items/{zotero_id}/file"
            f"?key={zotero_api_key}"
        )
        print(f"[DEBUG] Fetching PDF from URL: {file_url[:80]}...")
 
        # FIX: allow_redirects=True follows 302 redirects to cloud storage
        resp = requests.get(file_url, timeout=15, allow_redirects=True)
        print(f"[DEBUG] First request status: {resp.status_code}")
 
        if resp.status_code != 200:
            print(f"[DEBUG] Trying children of {zotero_id}")
            try:
                api = zotero.Zotero(zotero_library_id, zotero_library_type, zotero_api_key)
                children = api.children(zotero_id)
                print(f"[DEBUG] Found {len(children)} children")
                for child in children:
                    if (
                        child['data'].get('itemType') == 'attachment'
                        and child['data'].get('contentType') == 'application/pdf'
                    ):
                        print(f"[DEBUG] Found PDF attachment: {child['key']}")
                        child_file_url = (
                            f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}"
                            f"/items/{child['key']}/file?key={zotero_api_key}"
                        )
                        # FIX: allow_redirects=True on child request too
                        resp = requests.get(child_file_url, timeout=15, allow_redirects=True)
                        print(f"[DEBUG] Child request status: {resp.status_code}")
                        if resp.status_code == 200:
                            break
            except Exception as e:
                print(f"[DEBUG] Error checking children: {str(e)}")
 
        if resp.status_code != 200:
            print(f"[DEBUG] Failed to get PDF (status {resp.status_code})")
            return ""
 
        print(f"[DEBUG] Opening PDF and rendering page {page_num}")
        pdf_doc = fitz.open(stream=resp.content, filetype="pdf")
        print(f"[DEBUG] PDF has {len(pdf_doc)} pages")
 
        if page_num < 1 or page_num > len(pdf_doc):
            print(f"[DEBUG] Page {page_num} out of range, using page 1")
            page_num = 1
 
        page = pdf_doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        print(f"[DEBUG] Rendered pixmap: {pix.width}x{pix.height}")
 
        img_buffer = io.BytesIO()
        Image.open(io.BytesIO(pix.tobytes("png"))).save(img_buffer, format="PNG")
        b64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        print(f"[DEBUG] Encoded base64 image: {len(b64_img)} chars")
 
        pdf_doc.close()
        print(f"[DEBUG] Successfully created page image")
        return f"data:image/png;base64,{b64_img}"
    except Exception as e:
        print(f"[ERROR] Error fetching PDF page from Zotero: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
 
 
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
 
 
def get_item_url(item: dict) -> str:
    data = item.get('data', {})
    url = data.get('url', '') or data.get('uri', '')
    if url:
        return url
    links = item.get('links', {}) or {}
    alternate = links.get('alternate', {})
    if isinstance(alternate, dict):
        href = alternate.get('href', '')
        if href:
            return href
    return ''
 
 
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
 
 
def render_article_preview(relevant_chunks, show_debug=True):
    import sys
    debug_info = []
 
    debug_info.append(f"[DEBUG] render_article_preview called with {len(relevant_chunks) if relevant_chunks else 0} chunks")
    print(f"[DEBUG] render_article_preview called with {len(relevant_chunks) if relevant_chunks else 0} chunks", flush=True)
 
    if not relevant_chunks:
        st.info("No preview available for the current query.")
        return
 
    top_chunk = relevant_chunks[0]
    debug_info.append(f"[DEBUG] Top chunk keys: {list(top_chunk.keys())}")
    print(f"[DEBUG] Top chunk keys: {list(top_chunk.keys())}", flush=True)
 
    title = top_chunk.get('doc_title', 'Untitled')
    page = top_chunk.get('page')
    timestamp = top_chunk.get('timestamp')
    similarity = top_chunk.get('similarity', 0.0)
    preview_doc_id = top_chunk.get('doc_id')
 
    debug_info.append(f"[DEBUG] Extracted: title={title}, page={page}, preview_doc_id={preview_doc_id}, timestamp={timestamp}")
    print(f"[DEBUG] Extracted: title={title}, page={page}, preview_doc_id={preview_doc_id}, timestamp={timestamp}", flush=True)
 
    meta_parts = [f"Source: {title}"]
    if page:
        meta_parts.append(f"Page {page}")
    if timestamp:
        meta_parts.append(str(timestamp))
    meta_parts.append(f"Relevance: {similarity:.1%}")
 
    st.subheader("Article Preview")
    st.caption(" · ".join(meta_parts))
 
    same_doc_chunks = [chunk for chunk in relevant_chunks if chunk.get('doc_id') == preview_doc_id]
    same_doc_chunks = sorted(
        same_doc_chunks,
        key=lambda chunk: (chunk.get('chunk_position') is None, chunk.get('chunk_position', 0))
    )
 
    chunk_index = 0
    if top_chunk.get('chunk_position') is not None:
        for idx, chunk in enumerate(same_doc_chunks):
            if chunk.get('chunk_position') == top_chunk.get('chunk_position'):
                chunk_index = idx
                break
 
    start_idx = max(0, chunk_index - 1)
    end_idx = min(len(same_doc_chunks), chunk_index + 2)
    preview_chunks = same_doc_chunks[start_idx:end_idx]
 
    preview_sections = []
    for chunk in preview_chunks:
        chunk_title = []
        if chunk.get('chunk_position') is not None:
            chunk_title.append(f"Section {chunk['chunk_position']}")
        if chunk.get('page'):
            chunk_title.append(f"Page {chunk['page']}")
        if chunk.get('timestamp'):
            chunk_title.append(str(chunk['timestamp']))
        header = " | ".join(chunk_title) if chunk_title else ""
        if header:
            preview_sections.append(f"[{header}]\n{chunk.get('text', '').strip()}")
        else:
            preview_sections.append(chunk.get('text', '').strip())
 
    chunk_page_raw = top_chunk.get('chunk_page')
    debug_info.append(f"[DEBUG] chunk_page (raw from index): {chunk_page_raw}")
    print(f"[DEBUG] chunk_page (raw from index): {chunk_page_raw}", flush=True)
 
    if preview_doc_id:
        # ── Step 1: check stored page renders (free, no network call) ────────
        # At ingest time extract_pdf_page_previews() stores every page as a
        # base64 PNG in doc_image_urls with source="preview" and page=str(N).
        # Look for an exact match on chunk page before touching the network.
        stored_page_image = None
        target_page = str(page) if page else None
        all_doc_images = top_chunk.get("doc_image_urls") or []
 
        debug_info.append(f"[DEBUG] Searching {len(all_doc_images)} stored images for page {target_page}")
        print(f"[DEBUG] Searching {len(all_doc_images)} stored images for page {target_page}", flush=True)
 
        if target_page:
            for img_meta in all_doc_images:
                if (
                    isinstance(img_meta, dict)
                    and img_meta.get("source") == "preview"
                    and str(img_meta.get("page", "")) == target_page
                ):
                    stored_page_image = img_meta.get("data")
                    debug_info.append(f"[DEBUG] Found stored page render for page {target_page}")
                    print(f"[DEBUG] Found stored page render for page {target_page}", flush=True)
                    break
 
        # No exact page match — fall back to stored page 1
        if not stored_page_image:
            for img_meta in all_doc_images:
                if (
                    isinstance(img_meta, dict)
                    and img_meta.get("source") == "preview"
                    and str(img_meta.get("page", "")) == "1"
                ):
                    stored_page_image = img_meta.get("data")
                    debug_info.append(f"[DEBUG] No page {target_page} stored, using stored page 1 fallback")
                    print(f"[DEBUG] No page {target_page} stored, using stored page 1 fallback", flush=True)
                    break
 
        if stored_page_image:
            st.subheader("Source Page Preview")
            st.image(stored_page_image, use_container_width=True)
            caption = f"Page {target_page or '1'} from {title}"
            if not page:
                caption += " — rebuild the index to get the exact matched page"
            st.caption(caption)
            debug_info.append(f"[DEBUG] Displayed stored page render")
            print(f"[DEBUG] Displayed stored page render", flush=True)
            if show_debug:
                with st.expander("Debug Info"):
                    for line in debug_info:
                        st.code(line)
            return
 
        # ── Step 2: try Zotero API as last resort ─────────────────────────────
        # This path is hit only when the document was loaded before the
        # all-pages preview fix was applied (old ingest, no stored renders).
        try:
            page_num = int(page) if page else 1
            fallback_label = "" if page else " (no page found — showing page 1)"
            debug_info.append(f"[DEBUG] Calling fetch_pdf_page_from_zotero: doc_id={preview_doc_id}, page={page_num}{fallback_label}")
            print(f"[DEBUG] Calling fetch_pdf_page_from_zotero: doc_id={preview_doc_id}, page={page_num}{fallback_label}", flush=True)
 
            page_image = fetch_pdf_page_from_zotero(preview_doc_id, page_num)
            debug_info.append(f"[DEBUG] Got page_image back: {len(page_image) if page_image else 0} chars")
            print(f"[DEBUG] Got page_image back: {len(page_image) if page_image else 0} chars", flush=True)
 
            if page_image:
                st.subheader("Source Page Preview")
                st.image(page_image, use_container_width=True)
                caption_text = f"Page {page_num} from {title}"
                if not page:
                    caption_text += " — rebuild the index to get the exact matched page"
                st.caption(caption_text)
                debug_info.append(f"[DEBUG] Successfully displayed PDF page {page_num} via Zotero API")
                print(f"[DEBUG] Successfully displayed PDF page {page_num} via Zotero API", flush=True)
                if show_debug:
                    with st.expander("Debug Info"):
                        for line in debug_info:
                            st.code(line)
                return
            else:
                debug_info.append(f"[DEBUG] Zotero API also returned empty — falling back to text/image metadata")
                print(f"[DEBUG] Zotero API also returned empty — falling back to text/image metadata", flush=True)
        except Exception as e:
            debug_info.append(f"[ERROR] Could not fetch PDF page preview: {str(e)}")
            print(f"[ERROR] Could not fetch PDF page preview: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        debug_info.append(f"[DEBUG] No doc_id — cannot fetch PDF page")
        print(f"[DEBUG] No doc_id — cannot fetch PDF page", flush=True)
 
    # ── Step 3: fallback — show embedded/heuristic images from metadata ───────
    def _extract_image_data(image_meta):
        if isinstance(image_meta, dict):
            return image_meta.get('data') or image_meta.get('url') or image_meta.get('src')
        if isinstance(image_meta, str):
            return image_meta
        return None
 
    preview_images = []
    for chunk in preview_chunks:
        for image_meta in (chunk.get('chunk_image_urls') or chunk.get('doc_image_urls') or []):
            # Skip page_preview entries here — they were handled above
            if isinstance(image_meta, dict) and image_meta.get('source') == 'preview':
                continue
            image_data = _extract_image_data(image_meta)
            if image_data:
                preview_images.append((image_data, image_meta))
        if preview_images:
            break
 
    if preview_images:
        st.subheader("Preview Images")
        display_images = preview_images[:2]
        cols = st.columns(len(display_images))
 
        for col, (image_data, image_meta) in zip(cols, display_images):
            with col:
                if isinstance(image_data, str):
                    try:
                        st.image(image_data, use_container_width=True)
                    except Exception:
                        st.caption("Could not render the preview image.")
                else:
                    st.caption("Preview image data unavailable.")
 
                caption = ""
                if isinstance(image_meta, dict):
                    caption = (
                        image_meta.get('caption', '')
                        or image_meta.get('classification', '')
                        or image_meta.get('source', '')
                    )
                if caption:
                    st.caption(caption)
 
        if len(preview_images) > len(display_images):
            st.caption(f"{len(preview_images)} images available; showing first {len(display_images)}.")
        if show_debug:
            with st.expander("Debug Info"):
                for line in debug_info:
                    st.code(line)
        return
 
    # ── Step 4: last resort — plain text preview ──────────────────────────────
    preview_text = "\n\n---\n\n".join(preview_sections).strip()
    if not preview_text:
        preview_text = top_chunk.get('text', '').strip() or "No preview text available."
 
    st.text_area(
        "Preview",
        value=preview_text,
        height=320,
        key=f"preview_{preview_doc_id}",
        disabled=True
    )
 
    if show_debug:
        with st.expander("Debug Info"):
            for line in debug_info:
                st.code(line)
 
 
def remove_sources_block(text: str) -> str:
    pattern = r"(?s)\n{2}(?:---\n)?\*{0,2}Sources:\*{0,2}.*$"
    return re.sub(pattern, "", text).strip()
 
 
# ==================== APP SETUP ====================
st.set_page_config(page_title="Agile Biofoundry & ABPDU Query Tool", layout="wide")
 
_bootstrap_indexes()
 
st.title("Agile Biofoundry & ABPDU Query Tool")
 
zotero_library_id = _safe_secret("zotero_library_id")
zotero_api_key    = _safe_secret("zotero_api_key")
zotero_library_type = _safe_secret("zotero_library_type", "user")
openai_api_key = _safe_secret("openai_api_key")
 
client = OpenAI(api_key=openai_api_key) if openai_api_key else None
 
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
 
            filtered_indices = [
                i for i, m in enumerate(docs)
                if not search_term or search_term.lower() in m.get('title', '').lower()
            ]
            total_filtered = len(filtered_indices)
 
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
 
            st.divider()
            if st.button(
                "Rebuild Index",
                key=f"rebuild_idx_{collection_key}",
                use_container_width=True,
                help="Re-chunks and re-indexes stored articles. Use after updating the app to pick up [PAGE:N] page markers.",
            ):
                with st.spinner("Rebuilding index…"):
                    invalidate_index(collection_key)
                    documents, doc_ids_fresh, doc_metadata_fresh = get_all_articles(collection_key)
                    st.session_state[f"{prefix}_documents"]    = documents
                    st.session_state[f"{prefix}_doc_ids"]      = doc_ids_fresh
                    st.session_state[f"{prefix}_doc_metadata"] = doc_metadata_fresh
                    success = _build_index_for_collection(collection_key)
                    st.session_state["_indexes_bootstrapped"] = True
                if success:
                    st.success(f"Index rebuilt for {doc_count} documents.")
                else:
                    st.warning("No documents found to index. Load from Zotero first.")
                st.rerun()
 
 
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
                                item_url      = get_item_url(item)
                                zotero_id    = item['key']
 
                                if article_exists(zotero_id, None, collection_key):
                                    continue
                                if item_type in ['note', 'annotation']:
                                    continue
 
                                text = f"{title}\n{item['data'].get('abstractNote', '')}"
                                image_urls = []
 
                                if item_type != 'attachment':
                                    try:
                                        children = zot.children(zotero_id)
                                        for child in children:
                                            ctype = child['data'].get('contentType', '')
                                            if child['data'].get('itemType') == 'attachment':
                                                child_file_url = (
                                                    f"https://api.zotero.org/{zotero_library_type}s/"
                                                    f"{zotero_library_id}/items/{child['key']}/file?key={zotero_api_key}"
                                                )
                                                if ctype == 'application/pdf':
                                                    try:
                                                        resp = requests.get(child_file_url, timeout=15, allow_redirects=True)
                                                        if resp.status_code == 200:
                                                            text = f"{title}\n{extract_pdf_text(resp.content)}"
                                                            # FIX: no max_pages cap — store renders for every page
                                                            page_previews = extract_pdf_page_previews(resp.content)
                                                            image_urls.extend(page_previews)
                                                    except Exception:
                                                        pass
                                                elif ctype.startswith('image/'):
                                                    image_urls.append({
                                                        'data': child_file_url,
                                                        'source': 'attachment',
                                                        'classification': 'embedded',
                                                        'page': None,
                                                        'caption': ''
                                                    })
                                    except Exception:
                                        pass
 
                                if item_type == 'attachment':
                                    content_type = item['data'].get('contentType', '')
                                    file_url = (
                                        f"https://api.zotero.org/{zotero_library_type}s/"
                                        f"{zotero_library_id}/items/{zotero_id}/file?key={zotero_api_key}"
                                    )
 
                                    if content_type == 'application/pdf':
                                        try:
                                            resp = requests.get(file_url, timeout=15, allow_redirects=True)
                                            if resp.status_code == 200:
                                                text = f"{title}\n{extract_pdf_text(resp.content)}"
                                                # FIX: no max_pages cap — store renders for every page
                                                page_previews = extract_pdf_page_previews(resp.content)
                                                image_urls.extend(page_previews)
                                        except Exception:
                                            pass
                                    elif content_type.startswith('image/'):
                                        image_urls.append({
                                            'data': file_url,
                                            'source': 'attachment',
                                            'classification': 'embedded',
                                            'page': None,
                                            'caption': ''
                                        })
 
                                image_urls = dedupe_image_urls(image_urls)
 
                                success, _ = add_article(
                                    zotero_id, text, title, item_type,
                                    item['data'].get('abstractNote', ''),
                                    collection_key, low_relevance, item_url, image_urls=image_urls
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
 
        prompt = st.chat_input(
            f"Ask anything about {col_info['name']}...",
            key=f"chat_input_{collection_key}"
        )
 
        if prompt:
            st.session_state[f"{prefix}_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
 
            history = st.session_state[f"{prefix}_messages"]
            history_text = "\n".join(f"{msg['role']}:{msg['content']}" for msg in history)
            history_hash = hashlib.sha256(history_text.encode('utf-8')).hexdigest()
            cache_key = f"{collection_key}:{history_hash}"
 
            with st.chat_message("assistant"):
                if cache_key in st.session_state.query_cache:
                    result = st.session_state.query_cache[cache_key]
                    st.markdown(result)
                elif not index_ready(collection_key):
                    result = "No documents loaded yet. Please load from Zotero first."
                    st.markdown(result)
                elif not client:
                    result = "OpenAI API key not configured."
                    st.markdown(result)
                else:
                    relevant_chunks, seen_docs = retrieve_relevant_chunks(
                        prompt,
                        st.session_state[f"{prefix}_chunk_vectorizer"],
                        st.session_state[f"{prefix}_chunks"],
                        st.session_state[f"{prefix}_doc_id_mapping"],
                        faiss_index=st.session_state[f"{prefix}_faiss_index"],
                        k=st.session_state.get('k_chunks', 3),
                        similarity_threshold=0.12
                    )
 
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
                                "You are a helpful assistant."
                                " Base your answer only on the provided context."
                                " If the documents do not describe a collaboration or relationship, state that clearly but still summarize the most relevant related information."
                            )
                        }
                    ]
                    conversation.extend(previous_history)
                    conversation.append({
                        "role": "user",
                        "content": f"Context from documents:\n{context}\n\nQuestion: {latest_prompt}"
                    })
 
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=conversation,
                        stream=True,
                        max_tokens=1024,
                    )
 
                    result = st.write_stream(
                        chunk.choices[0].delta.content or "" for chunk in stream
                    )
                    result = remove_sources_block(result or "")
 
                    if cited_docs:
                        sources = "\n\n---\n**Sources:**\n"
                        for doc in cited_docs:
                            title = doc['title']
                            url = doc.get('url', '')
                            status = doc.get('image_status', 'none')
                            status_label = {
                                'embedded': 'embedded image',
                                'heuristic': 'heuristic image',
                                'none': ''
                            }.get(status, '')
                            if url:
                                if status_label:
                                    sources += f"- [{title}]({url}) (Relevance: {doc['similarity']:.1%}; {status_label})\n"
                                else:
                                    sources += f"- [{title}]({url}) (Relevance: {doc['similarity']:.1%})\n"
                            else:
                                if status_label:
                                    sources += f"- {title} (Relevance: {doc['similarity']:.1%}; {status_label})\n"
                                else:
                                    sources += f"- {title} (Relevance: {doc['similarity']:.1%})\n"
                        st.markdown(sources)
                        result = (result or "") + sources
 
                        render_article_preview(relevant_chunks)
 
with st.bottom:
    st.caption("Zotero Library Source: https://www.zotero.org/groups/6420515/abpdu_workflow_automation-article_query_tool/collections/LRILZKMS/collection")
 