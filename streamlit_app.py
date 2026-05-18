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
from streamlit_extras.bottom_container import bottom

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
try:
    from PIL import Image, ImageFilter
    import numpy as np
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Image processing libraries not available: {e}")
    Image = None
    np = None
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


def extract_pdf_images(pdf_content, max_images: int = 3):
    """Extract embedded images from PDF using PyMuPDF (fitz)."""
    if not FITZ_AVAILABLE:
        return []

    try:
        pdf_doc = fitz.open(stream=pdf_content, file_type="pdf")
        images_data = []
        seen_xrefs = set()

        for page_num in range(len(pdf_doc)):
            if len(images_data) >= max_images:
                break

            page = pdf_doc[page_num]
            image_list = page.get_images(full=True)

            for img_index in image_list:
                if len(images_data) >= max_images:
                    break

                xref = img_index[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    pix = fitz.Pixmap(pdf_doc, xref)
                    if pix.n - pix.alpha < 4:
                        timage = fitz.Pixmap(fitz.csRGB, pix)
                    else:
                        timage = pix

                    img_bytes = timage.tobytes("png")
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    images_data.append({
                        'data': f"data:image/png;base64,{b64_img}",
                        'page': page_num + 1,
                        'source': 'embedded',
                        'classification': 'embedded',
                        'caption': ''
                    })
                except Exception:
                    continue

        pdf_doc.close()
        return images_data
    except Exception as e:
        print(f"Error extracting images: {str(e)}")
        return []


def _extract_caption_from_page(page, bbox, zoom=2):
    try:
        captions = []
        x0, y0, x1, y1 = bbox
        for block in page.get_text("blocks") or []:
            bx0, by0, bx1, by1, text = block[:5]
            if not isinstance(text, str) or not text.strip():
                continue
            bx0, by0, bx1, by1 = bx0 * zoom, by0 * zoom, bx1 * zoom, by1 * zoom
            if by0 >= y1 and by0 <= y1 + max(80, int((y1 - y0) * 0.75)) and not (bx1 < x0 or bx0 > x1):
                captions.append(text.strip())

        for caption in captions:
            if re.search(r"\bfig(?:ure)?\b", caption, re.I):
                return caption
        return captions[0] if captions else ""
    except Exception:
        return ""


def _region_looks_like_image(page, bbox, zoom=2, edge_mask=None):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if width < 40 or height < 40:
        return False

    if edge_mask is not None:
        region = edge_mask[y0:y1, x0:x1]
        density = float(region.sum()) / max(1, width * height)
        if density < 0.002:
            return False

    overlap = 0
    region_area = width * height
    for block in page.get_text("blocks") or []:
        bx0, by0, bx1, by1, text = block[:5]
        if not isinstance(text, str) or not text.strip():
            continue
        bx0, by0, bx1, by1 = bx0 * zoom, by0 * zoom, bx1 * zoom, by1 * zoom
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        overlap_x = max(0, min(x1, bx1) - max(x0, bx0))
        overlap_y = max(0, min(y1, by1) - max(y0, by0))
        overlap += overlap_x * overlap_y

    if region_area > 0 and overlap / region_area > 0.4:
        return False

    return True


def _find_connected_components(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            y0 = y1 = y
            x0 = x1 = x
            count = 0

            while stack:
                cy, cx = stack.pop()
                count += 1
                y0 = min(y0, cy)
                y1 = max(y1, cy)
                x0 = min(x0, cx)
                x1 = max(x1, cx)

                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append((x0, y0, x1 + 1, y1 + 1, count))

    return components


def _find_pdf_image_block_regions(page, max_regions: int = 3, zoom: int = 2):
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return []

    page_area = page.rect.width * page.rect.height
    candidates = []
    for block in blocks:
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox", [])
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(v * zoom) for v in bbox]
        width = x1 - x0
        height = y1 - y0
        area = width * height
        if area < page_area * 0.01 or area > page_area * 0.7:
            continue
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 6:
            continue
        candidates.append((x0, y0, x1, y1))
        if len(candidates) >= max_regions:
            break
    return candidates


def _find_image_regions_on_page(pil_img, page, max_regions: int = 3, zoom: int = 2):
    block_regions = _find_pdf_image_block_regions(page, max_regions=max_regions, zoom=zoom)
    if block_regions:
        return [(x0, y0, x1, y1, 'embedded') for x0, y0, x1, y1 in block_regions]

    gray = pil_img.convert("L")
    edge_image = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edge_image, dtype=np.uint8)
    mask = edge_arr > 15

    components = _find_connected_components(mask)
    page_area = mask.shape[0] * mask.shape[1]
    candidates = []

    for x0, y0, x1, y1, count in sorted(components, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]), reverse=True):
        width = x1 - x0
        height = y1 - y0
        area = width * height
        if area < page_area * 0.015 or area > page_area * 0.6:
            continue
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 6:
            continue
        if not _region_looks_like_image(page, (x0, y0, x1, y1), zoom=zoom, edge_mask=mask):
            continue
        candidates.append((x0, y0, x1, y1, 'heuristic'))
        if len(candidates) >= max_regions:
            break

    return candidates


def classify_image_region(pil_img):
    """Classify an extracted image region based on its grayscale and color properties."""
    if pil_img is None:
        return "unknown"

    width, height = pil_img.size
    aspect_ratio = width / height if height > 0 else 1
    hsv = np.array(pil_img.convert("HSV"), dtype=np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue_std = np.std(hue)
    sat_mean = np.mean(sat)
    val_mean = np.mean(val)

    if aspect_ratio > 2:
        return "chart" if sat_mean > 80 else "diagram"
    if aspect_ratio < 0.5:
        return "graph"
    if hue_std > 30 and sat_mean > 70:
        return "color_figure"
    if val_mean < 100:
        return "microscopy"
    return "figure"


def detect_colorful_rectangles(pdf_content, max_images: int = 3):
    """Extract embedded images first, then use rendered rectangle heuristics as a fallback."""
    if not FITZ_AVAILABLE:
        return []

    try:
        extracted_images = extract_pdf_images(pdf_content, max_images=max_images)
        remaining = max_images - len(extracted_images)
        if remaining <= 0:
            return extracted_images

        pdf_doc = fitz.open(stream=pdf_content, file_type="pdf")
        for page_num in range(len(pdf_doc)):
            if len(extracted_images) >= max_images:
                break

            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

            regions = _find_image_regions_on_page(pil_img, page, max_regions=remaining, zoom=2)
            for x0, y0, x1, y1, source in regions:
                if len(extracted_images) >= max_images:
                    break

                region = pil_img.crop((max(0, x0 - 5), max(0, y0 - 5), min(pil_img.width, x1 + 5), min(pil_img.height, y1 + 5)))
                region_arr = np.array(region.convert("L"), dtype=np.float32)
                if region_arr.std() < 20:
                    continue

                caption = _extract_caption_from_page(page, (x0, y0, x1, y1), zoom=2)
                buffer = io.BytesIO()
                region.save(buffer, format="PNG")
                b64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
                extracted_images.append({
                    'data': f"data:image/png;base64,{b64_img}",
                    'classification': source,
                    'page': page_num + 1,
                    'source': source,
                    'caption': caption
                })
                remaining -= 1

        if not extracted_images:
            page = pdf_doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            buffer = io.BytesIO()
            Image.open(io.BytesIO(pix.tobytes("png"))).save(buffer, format="PNG")
            b64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            extracted_images.append({
                'data': f"data:image/png;base64,{b64_img}",
                'classification': 'page_preview',
                'page': 1,
                'source': 'preview',
                'caption': ''
            })

        pdf_doc.close()
        return extracted_images
    except Exception as e:
        print(f"Error detecting colorful rectangles: {str(e)}")
        return []


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


def render_article_preview(relevant_chunks):
    if not relevant_chunks:
        st.info("No preview available for the current query.")
        return

    top_chunk = relevant_chunks[0]
    title = top_chunk.get('doc_title', 'Untitled')
    page = top_chunk.get('page')
    timestamp = top_chunk.get('timestamp')
    similarity = top_chunk.get('similarity', 0.0)

    meta_parts = [f"Source: {title}"]
    if page:
        meta_parts.append(f"Page {page}")
    if timestamp:
        meta_parts.append(str(timestamp))
    meta_parts.append(f"Relevance: {similarity:.1%}")

    st.subheader("Article Preview")
    st.caption(" · ".join(meta_parts))

    preview_doc_id = top_chunk.get('doc_id')
    preview_chunks = [chunk for chunk in relevant_chunks if chunk.get('doc_id') == preview_doc_id]
    preview_texts = [chunk.get('text', '').strip() for chunk in preview_chunks if chunk.get('text')]
    preview_text = "\n\n---\n\n".join(preview_texts) if preview_texts else top_chunk.get('text', '')

    st.text_area(
        "Preview",
        value=preview_text,
        height=320,
        key=f"preview_{preview_doc_id}",
        disabled=True
    )


def remove_sources_block(text: str) -> str:
    # Remove any model-generated trailing sources block to prevent duplication.
    import re
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
                                                        resp = requests.get(child_file_url, timeout=15)
                                                        if resp.status_code == 200:
                                                            text = f"{title}\n{extract_pdf_text(resp.content)}"
                                                            image_urls = detect_colorful_rectangles(resp.content, max_images=3)
                                                    except Exception:
                                                        pass
                                                elif ctype.startswith('image/'):
                                                    image_urls.append(child_file_url)
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
                                            resp = requests.get(file_url, timeout=15)
                                            if resp.status_code == 200:
                                                text = f"{title}\n{extract_pdf_text(resp.content)}"
                                                image_urls = detect_colorful_rectangles(resp.content, max_images=3)
                                        except Exception:
                                            pass
                                    elif content_type.startswith('image/'):
                                        image_urls.append({
                                            'data': file_url,
                                            'source': 'embedded',
                                            'classification': 'embedded',
                                            'page': None,
                                        })

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
                                'none': 'no image'
                            }.get(status, 'no image')
                            if url:
                                sources += f"- [{title}]({url}) (Relevance: {doc['similarity']:.1%}; {status_label})\n"
                            else:
                                sources += f"- {title} (Relevance: {doc['similarity']:.1%}; {status_label})\n"
                        st.markdown(sources)
                        result = (result or "") + sources

                        render_article_preview(relevant_chunks)

with bottom():
    st.caption("Zotero Library Source: https://www.zotero.org/groups/6420515/abpdu_workflow_automation-article_query_tool/collections/LRILZKMS/collection")