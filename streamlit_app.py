import streamlit as st
from pyzotero import zotero
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import PyPDF2
import requests
import io
import os
import json

# ==================== CONFIG ====================
COLLECTIONS = [
    {"name": "Agile BioFoundry", "key": "agile", "secret": "zotero_collection_agile"},
    {"name": "ABPDU", "key": "abpdu", "secret": "zotero_collection_abpdu"}
]

# ==================== HELPERS ====================

def extract_pdf_text(pdf_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"


def is_item_low_relevance(item):
    tags = item.get('data', {}).get('tags', []) or []
    for tag in tags:
        tag_value = tag.get('tag', '') if isinstance(tag, dict) else str(tag)
        normalized = tag_value.strip().lower().replace(' ', '-')
        if normalized in ['low-relevance', 'lowrelevance', 'low relevance']:
            return True
    return False


def init_collection_state(collection_key):
    keys = [
        f"documents_{collection_key}",
        f"doc_ids_{collection_key}",
        f"doc_metadata_{collection_key}",
        f"vectorizer_{collection_key}",
        f"tfidf_matrix_{collection_key}",
        f"messages_{collection_key}"
    ]
    defaults = [[], [], [], None, None, []]
    for key, default in zip(keys, defaults):
        if key not in st.session_state:
            st.session_state[key] = default


def set_collection_state(collection_key, documents=None, doc_ids=None, doc_metadata=None, vectorizer=None, tfidf_matrix=None, messages=None):
    if documents is not None:
        st.session_state[f"documents_{collection_key}"] = documents
    if doc_ids is not None:
        st.session_state[f"doc_ids_{collection_key}"] = doc_ids
    if doc_metadata is not None:
        st.session_state[f"doc_metadata_{collection_key}"] = doc_metadata
    if vectorizer is not None:
        st.session_state[f"vectorizer_{collection_key}"] = vectorizer
    if tfidf_matrix is not None:
        st.session_state[f"tfidf_matrix_{collection_key}"] = tfidf_matrix
    if messages is not None:
        st.session_state[f"messages_{collection_key}"] = messages


def load_zotero_collection(collection_key, collection_secret_key):
    if not zotero_library_id or not zotero_api_key:
        st.error("Zotero Library ID and API Key must be set in Streamlit secrets.")
        return

    with st.spinner(f"Loading {collection_key} documents from Zotero..."):
        try:
            zot = zotero.Zotero(zotero_library_id, zotero_library_type, zotero_api_key)
            collection_key_value = st.secrets.get(collection_secret_key, "")

            if collection_key_value:
                items = zot.everything(zot.collection_items(collection_key_value))
            else:
                items = zot.everything(zot.items())

            documents = []
            doc_ids = []
            doc_metadata = []

            total_items = len(items)
            progress = st.progress(0)

            for item_idx, item in enumerate(items, start=1):
                progress.progress(item_idx / (total_items or 1))

                if item['data']['itemType'] not in ['journalArticle', 'webpage', 'report', 'conferencePaper', 'videoRecording', 'audioRecording', 'book', 'bookSection', 'preprint', 'document', 'presentation']:
                    continue

                title = item['data'].get('title', 'Untitled')
                abstract = item['data'].get('abstractNote', '')
                item_type = item['data']['itemType']

                notes = []
                try:
                    children = zot.everything(zot.children(item['key']))
                except Exception:
                    children = []

                for child in children:
                    if child['data']['itemType'] == 'note':
                        notes.append(child['data'].get('note', ''))

                text = f"{title}\n{abstract}\n{' '.join(notes)}"

                for child in children:
                    if child['data']['itemType'] == 'attachment':
                        link_mode = child['data'].get('linkMode', '')
                        if link_mode in ['linked_file', 'imported_file', 'imported_url']:
                            file_url = f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}/items/{child['key']}/file?key={zotero_api_key}"
                            try:
                                response = requests.get(file_url, timeout=10)
                                if response.status_code == 200:
                                    content_type = response.headers.get('Content-Type', '')
                                    if 'application/pdf' in content_type:
                                        pdf_text = extract_pdf_text(response.content)
                                        text += f"\n{pdf_text}"
                                    elif 'text/html' in content_type:
                                        text += f"\n{response.text}"
                            except Exception:
                                pass

                if text.strip():
                    low_relevance = is_item_low_relevance(item)
                    documents.append(text)
                    doc_ids.append(item['key'])
                    doc_metadata.append({
                        'title': title,
                        'itemType': item_type,
                        'abstract': abstract[:200] if abstract else '',
                        'low_relevance': low_relevance
                    })

            progress.empty()

            if not documents:
                st.error("No documents found in Zotero for this collection.")
                return

            set_collection_state(collection_key, documents=documents, doc_ids=doc_ids, doc_metadata=doc_metadata)
            if documents:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                set_collection_state(collection_key, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)

            st.success(f"Loaded {len(documents)} documents for {collection_key}.")
            st.experimental_rerun()

        except Exception as e:
            st.error(f"Error loading Zotero collection: {e}")


# ==================== PAGE SETUP ====================
st.set_page_config(page_title="Agile Biofoundry & ABPDU Query", layout="wide")
st.title("Agile Biofoundry & ABPDU Query Tool")

# Load secrets
zotero_library_id = st.secrets.get("zotero_library_id", "")
zotero_api_key = st.secrets.get("zotero_api_key", "")
zotero_library_type = st.secrets.get("zotero_library_type", "user")
openai_api_key = st.secrets.get("openai_api_key", "")

# openai client
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# session initialization
for c in COLLECTIONS:
    init_collection_state(c['key'])

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Collections")
    for c in COLLECTIONS:
        docs = st.session_state[f"documents_{c['key']}"]
        st.write(f"- {c['name']}: {len(docs)} docs")

    st.write("---")
    st.write("Tag rule: set Zotero tag 'low-relevance' to lower a doc's weight in retrieval.")

# ==================== TAB UI ====================
collection_tabs = st.tabs([c['name'] for c in COLLECTIONS])

for tab, c in zip(collection_tabs, COLLECTIONS):
    with tab:
        col_key = c['key']
        col_state = {
            'documents': st.session_state[f"documents_{col_key}"],
            'doc_ids': st.session_state[f"doc_ids_{col_key}"],
            'doc_metadata': st.session_state[f"doc_metadata_{col_key}"],
            'vectorizer': st.session_state[f"vectorizer_{col_key}"],
            'tfidf_matrix': st.session_state[f"tfidf_matrix_{col_key}"],
            'messages': st.session_state[f"messages_{col_key}"]
        }

        st.subheader(f"{c['name']}")

        if st.button("Load Zotero Library", key=f"load_{col_key}"):
            load_zotero_collection(col_key, c['secret'])

        st.write(f"Documents loaded: {len(col_state['documents'])}")

        if col_state['documents'] and col_state['vectorizer'] is not None and col_state['tfidf_matrix'] is not None:
            # show loaded docs summary
            if col_state['messages']:
                for msg in col_state['messages']:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])

            if prompt := st.chat_input(f"Ask about {c['name']}:", key=f"prompt_{col_key}"):
                messages = col_state['messages'] + [{"role": "user", "content": prompt}]
                set_collection_state(col_key, messages=messages)

                query_vec = col_state['vectorizer'].transform([prompt])
                similarities = cosine_similarity(query_vec, col_state['tfidf_matrix']).flatten()
                weighted = similarities.copy()

                for i, meta in enumerate(col_state['doc_metadata']):
                    if meta.get('low_relevance'):
                        weighted[i] *= 0.5

                top_indices = np.argsort(weighted)[-3:][::-1]

                context = ""
                cited_docs = []
                for idx in top_indices:
                    if weighted[idx] > 0.1:
                        context += f"\n\nDocument ID: {col_state['doc_ids'][idx]}\n{col_state['documents'][idx][:1000]}..."
                        cited_docs.append({
                            'title': col_state['doc_metadata'][idx].get('title', 'Untitled'),
                            'id': col_state['doc_ids'][idx],
                            'similarity': float(weighted[idx]),
                            'low_relevance': col_state['doc_metadata'][idx].get('low_relevance', False)
                        })

                if not context:
                    context = "No relevant documents found."

                if not client:
                    response = "OpenAI API key not set in secrets."
                else:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": f"You are a helpful assistant for {c['name']}. Use the context to answer."},
                                {"role": "user", "content": f"Context: {context}\n\nQuery: {prompt}"}
                            ]
                        ).choices[0].message.content

                        if cited_docs:
                            response += "\n\n---\n**Sources:**\n"
                            for doc in cited_docs:
                                extra = " (low relevance applied)" if doc.get('low_relevance') else ""
                                response += f"- {doc['title']} (ID: {doc['id']}, Relevance: {doc['similarity']:.2%}){extra}\n"
                    except Exception as e:
                        response = f"Error generating response: {e}"

                set_collection_state(col_key, messages=col_state['messages'] + [{"role": "assistant", "content": response}])
                st.experimental_rerun()
        else:
            st.info("Load documents and wait for index before querying.")
