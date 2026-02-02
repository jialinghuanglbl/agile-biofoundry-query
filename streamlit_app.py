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
from article_storage import (
    load_articles,
    save_articles,
    article_exists,
    add_article,
    get_all_articles,
    remove_article,
    clear_all_articles,
    get_article_count,
    rename_article
)
from document_retrieval import (
    create_chunked_documents,
    retrieve_relevant_chunks,
    format_context_from_chunks
)

# Function to extract text from PDF content
def extract_pdf_text(pdf_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"


def ensure_chunk_index() -> bool:
    """Ensure that chunk index and TF-IDF are initialized in session state.

    Returns True if an index is available, False otherwise.
    """
    # If already present and non-empty, we're good
    if st.session_state.get('chunk_vectorizer') is not None and st.session_state.get('chunk_tfidf_matrix') is not None and st.session_state.get('chunks'):
        return True

    # Need documents to build index
    if 'documents' not in st.session_state or not st.session_state.documents:
        return False

    chunk_size = st.session_state.get('chunk_size', 800)
    overlap = st.session_state.get('overlap', 150)
    summary_sentences = st.session_state.get('summary_sentences', 3)
    use_summaries = st.session_state.get('use_summaries', True)

    chunks, doc_id_mapping = create_chunked_documents(
        st.session_state.documents,
        st.session_state.doc_ids,
        st.session_state.doc_metadata,
        chunk_size=chunk_size,
        overlap=overlap,
        summary_sentences=summary_sentences
    )

    if not chunks:
        # No chunks could be created
        st.session_state.chunks = []
        st.session_state.doc_id_mapping = []
        return False

    st.session_state.chunks = chunks
    st.session_state.doc_id_mapping = doc_id_mapping

    chunk_texts = [c['summary'] if use_summaries and c.get('summary') else c['text'] for c in chunks]
    chunk_vectorizer = TfidfVectorizer(stop_words='english')
    chunk_tfidf_matrix = chunk_vectorizer.fit_transform(chunk_texts)
    st.session_state.chunk_vectorizer = chunk_vectorizer
    st.session_state.chunk_tfidf_matrix = chunk_tfidf_matrix

    st.success(f"Auto-rebuilt index with {len(chunks)} chunks")
    return True

# Streamlit app
st.title("Agile Biofoundry Zotero Query App")

# Load credentials from secrets
zotero_library_id = st.secrets["zotero_library_id"]
zotero_api_key = st.secrets["zotero_api_key"]
zotero_library_type = st.secrets["zotero_library_type"]
openai_api_key = st.secrets["openai_api_key"]
zotero_collection_key = st.secrets.get("zotero_collection_key", "")  # Optional

# Initialize OpenAI client AFTER loading the API key
client = OpenAI(api_key=openai_api_key)

# Initialize session state for documents
if "documents" not in st.session_state:
    # Load from persistent storage
    documents, doc_ids, doc_metadata = get_all_articles()
    st.session_state.documents = documents
    st.session_state.doc_ids = doc_ids
    st.session_state.doc_metadata = doc_metadata
    
    # Fit TF-IDF if documents exist
    if documents:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix
        
        # Build chunks using current settings
        chunk_size = st.session_state.get('chunk_size', 800)
        overlap = st.session_state.get('overlap', 150)
        summary_sentences = st.session_state.get('summary_sentences', 3)
        use_summaries = st.session_state.get('use_summaries', True)

        chunks, doc_id_mapping = create_chunked_documents(
            documents, doc_ids, doc_metadata,
            chunk_size=chunk_size,
            overlap=overlap,
            summary_sentences=summary_sentences
        )
        st.session_state.chunks = chunks
        st.session_state.doc_id_mapping = doc_id_mapping
        
        # Use summaries or full text for chunk indexing
        chunk_texts = [c['summary'] if use_summaries else c['text'] for c in chunks]
        chunk_vectorizer = TfidfVectorizer(stop_words='english')
        chunk_tfidf_matrix = chunk_vectorizer.fit_transform(chunk_texts)
        st.session_state.chunk_vectorizer = chunk_vectorizer
        st.session_state.chunk_tfidf_matrix = chunk_tfidf_matrix
    
if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = []

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Retrieval settings (fixed)
    st.subheader("Retrieval Settings")

    # Ensure defaults are set in session state (immutable by UI)
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 800
    if 'overlap' not in st.session_state:
        st.session_state.overlap = 150
    if 'summary_sentences' not in st.session_state:
        st.session_state.summary_sentences = 3
    if 'use_summaries' not in st.session_state:
        st.session_state.use_summaries = True
    if 'k_chunks' not in st.session_state:
        st.session_state.k_chunks = 6

    # Display document count
    if st.session_state.documents:
        st.metric("Total Documents", len(st.session_state.documents))
        
        # View documents section
        st.subheader("View Documents")
        
        # Search/filter documents
        search_term = st.text_input("Search documents", "")
        
        # Filter documents based on search
        filtered_indices = []
        for idx, metadata in enumerate(st.session_state.doc_metadata):
            title = metadata.get('title', 'Untitled')
            if search_term.lower() in title.lower():
                filtered_indices.append(idx)
        
        if not search_term:
            filtered_indices = list(range(len(st.session_state.doc_metadata)))
        
        st.write(f"Showing {len(filtered_indices)} of {len(st.session_state.documents)} documents")
        
        # Display each document with options
        for idx in filtered_indices:
            metadata = st.session_state.doc_metadata[idx]
            title = metadata.get('title', 'Untitled')
            item_type = metadata.get('itemType', 'Unknown')
            
            with st.expander(f"{title[:50]}{'...' if len(title) > 50 else ''}"):
                st.write(f"**Type:** {item_type}")
                st.write(f"**Zotero ID:** {st.session_state.doc_ids[idx]}")
                
                # Show preview of content
                preview = st.session_state.documents[idx][:300]
                st.text_area("Preview", preview, height=100, disabled=True, key=f"preview_{idx}")
                
                # Rename controls
                rename_default = st.session_state.doc_metadata[idx].get('title', 'Untitled')
                new_title = st.text_input("Rename title", value=rename_default, key=f"rename_input_{idx}")
                if st.button("Rename", key=f"rename_{idx}"):
                    zotero_id = st.session_state.doc_ids[idx]
                    success, msg = rename_article(zotero_id, new_title)
                    if success:
                        st.session_state.doc_metadata[idx]['title'] = new_title
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

                # Delete button
                if st.button(f"Delete", key=f"delete_{idx}"):
                    # Remove from storage
                    zotero_id = st.session_state.doc_ids[idx]
                    remove_article(zotero_id)
                    
                    # Remove from session state
                    del st.session_state.documents[idx]
                    del st.session_state.doc_ids[idx]
                    del st.session_state.doc_metadata[idx]
                    
                    # Refit TF-IDF and chunks if documents remain
                    if st.session_state.documents:
                        vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(st.session_state.documents)
                        st.session_state.vectorizer = vectorizer
                        st.session_state.tfidf_matrix = tfidf_matrix
                        
                        # Recreate chunks
                        chunks, doc_id_mapping = create_chunked_documents(
                            st.session_state.documents,
                            st.session_state.doc_ids,
                            st.session_state.doc_metadata
                        )
                        st.session_state.chunks = chunks
                        st.session_state.doc_id_mapping = doc_id_mapping
                        
                        use_summaries = st.session_state.get('use_summaries', True)
                        chunk_texts = [c['summary'] if use_summaries and c.get('summary') else c['text'] for c in chunks]
                        chunk_vectorizer = TfidfVectorizer(stop_words='english')
                        chunk_tfidf_matrix = chunk_vectorizer.fit_transform(chunk_texts)
                        st.session_state.chunk_vectorizer = chunk_vectorizer
                        st.session_state.chunk_tfidf_matrix = chunk_tfidf_matrix
                        # Notify user that index was auto-rebuilt after deletion
                        st.success(f"Auto-rebuilt index with {len(chunks)} chunks after deletion")
                    else:
                        # Clear vectorizer and chunks if no documents
                        if 'vectorizer' in st.session_state:
                            del st.session_state.vectorizer
                        if 'tfidf_matrix' in st.session_state:
                            del st.session_state.tfidf_matrix
                        if 'chunks' in st.session_state:
                            del st.session_state.chunks
                        if 'doc_id_mapping' in st.session_state:
                            del st.session_state.doc_id_mapping
                        if 'chunk_vectorizer' in st.session_state:
                            del st.session_state.chunk_vectorizer
                        if 'chunk_tfidf_matrix' in st.session_state:
                            del st.session_state.chunk_tfidf_matrix
                        # Inform user that documents were cleared and index rebuilt (empty)
                        st.success("Cleared all documents and rebuilt an empty index")
                    
                    st.rerun()
        
        # Bulk actions
        st.subheader("Bulk Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear All", type="secondary"):
                clear_all_articles()
                st.session_state.documents = []
                st.session_state.doc_ids = []
                st.session_state.doc_metadata = []
                if 'vectorizer' in st.session_state:
                    del st.session_state.vectorizer
                if 'tfidf_matrix' in st.session_state:
                    del st.session_state.tfidf_matrix
                if 'chunks' in st.session_state:
                    del st.session_state.chunks
                if 'doc_id_mapping' in st.session_state:
                    del st.session_state.doc_id_mapping
                if 'chunk_vectorizer' in st.session_state:
                    del st.session_state.chunk_vectorizer
                if 'chunk_tfidf_matrix' in st.session_state:
                    del st.session_state.chunk_tfidf_matrix
                # Inform user about auto-rebuild (empty index)
                st.success("Cleared all documents and rebuilt an empty index")
                st.rerun()
        
        with col2:
            # Export metadata as JSON
            if st.button("Export List"):
                export_data = {
                    "count": len(st.session_state.documents),
                    "documents": [
                        {
                            "zotero_id": st.session_state.doc_ids[i],
                            "title": st.session_state.doc_metadata[i].get('title', 'Untitled'),
                            "type": st.session_state.doc_metadata[i].get('itemType', 'Unknown')
                        }
                        for i in range(len(st.session_state.documents))
                    ]
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="zotero_documents.json",
                    mime="application/json"
                )
    else:
        st.info("No documents loaded yet. Click 'Load Zotero Library' to get started.")

# Main content area
# Button to load library
if st.button("Load Zotero Library", type="primary"):
    if not zotero_library_id or not zotero_api_key:
        st.error("Zotero Library ID and API Key must be set in Streamlit secrets.")
    else:
        with st.spinner("Loading documents from Zotero..."):
            try:
                zot = zotero.Zotero(zotero_library_id, zotero_library_type, zotero_api_key)

                # Fetch items from collection if key provided, else all items
                if zotero_collection_key:
                    items = zot.collection_items(zotero_collection_key)
                else:
                    items = zot.items()

                documents = []
                doc_ids = []
                doc_metadata = []
                duplicates = []
                new_count = 0
                
                # Initialize skip tracking
                skipped_items = {
                    'wrong_type': [],
                    'duplicate': [],
                    'no_content': [],
                    'errors': []
                }

                progress_bar = st.progress(0)
                total_items = len(items)

                for item_idx, item in enumerate(items):
                    # Update progress
                    progress_bar.progress((item_idx + 1) / total_items)
                    
                    item_type = item['data']['itemType']
                    title = item['data'].get('title', 'Untitled')
                    
                    # Skip notes and annotations (but NOT attachments - we'll process those)
                    if item_type in ['note', 'annotation']:
                        continue
                    
                    # Check if article already exists
                    if article_exists(item['key'], load_articles()):
                        duplicates.append(title)
                        skipped_items['duplicate'].append(title)
                        continue

                    try:
                        # Handle standalone attachments (PDFs without parent items)
                        if item_type == 'attachment':
                            link_mode = item['data'].get('linkMode', '')
                            content_type = item['data'].get('contentType', '')
                            
                            # Only process PDF attachments
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
                                                
                                                # Add standalone PDF
                                                success, msg = add_article(item['key'], text, title, 'attachment', '')
                                                if success:
                                                    documents.append(text)
                                                    doc_ids.append(item['key'])
                                                    doc_metadata.append({
                                                        'title': title,
                                                        'itemType': 'attachment (PDF)',
                                                        'abstract': ''
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
                        
                        # Track items with other non-supported types
                        if item_type not in ['journalArticle', 'webpage', 'report', 'conferencePaper', 'book', 'bookSection', 'preprint', 'document', 'presentation']:
                            skipped_items['wrong_type'].append(f"{title} ({item_type})")
                            continue

                        # Extract metadata text for regular items
                        abstract = item['data'].get('abstractNote', '')
                        notes = []

                        # Get child notes
                        children = zot.children(item['key'])
                        for child in children:
                            if child['data']['itemType'] == 'note':
                                notes.append(child['data'].get('note', ''))

                        text = f"{title}\n{abstract}\n{' '.join(notes)}"

                        # If there are attachments (e.g., PDF or snapshot)
                        pdf_extracted = False
                        for child in children:
                            if child['data']['itemType'] == 'attachment':
                                link_mode = child['data'].get('linkMode', '')
                                if link_mode in ['linked_file', 'imported_file', 'imported_url']:
                                    # Get the file URL via API
                                    file_url = f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}/items/{child['key']}/file?key={zotero_api_key}"
                                    try:
                                        response = requests.get(file_url, timeout=10)
                                        if response.status_code == 200:
                                            content_type = response.headers.get('Content-Type', '')
                                            if 'application/pdf' in content_type:
                                                pdf_text = extract_pdf_text(response.content)
                                                if not pdf_text.startswith("Error"):
                                                    text += f"\n{pdf_text}"
                                                    pdf_extracted = True
                                            elif 'text/html' in content_type:
                                                text += f"\n{response.text}"
                                    except Exception as attach_err:
                                        # Log but don't fail the entire item
                                        pass

                        if text.strip():
                            # Add to storage
                            success, msg = add_article(item['key'], text, title, item_type, abstract)
                            if success:
                                documents.append(text)
                                doc_ids.append(item['key'])
                                doc_metadata.append({
                                    'title': title,
                                    'itemType': item_type,
                                    'abstract': abstract[:200] if abstract else ''
                                })
                                new_count += 1
                        else:
                            skipped_items['no_content'].append(title)
                            
                    except Exception as e:
                        skipped_items['errors'].append(f"{title}: {str(e)}")

                progress_bar.empty()

                # Store skip report in session state for persistence
                st.session_state.last_load_report = {
                    'new_count': new_count,
                    'duplicates': len(duplicates),
                    'skipped_items': skipped_items,
                    'total_processed': total_items
                }

                if not documents and not duplicates:
                    st.error("No documents found in the library.")
                else:
                    # Reload ALL documents from storage (including previously stored ones)
                    all_documents, all_doc_ids, all_doc_metadata = get_all_articles()
                    
                    # Store all documents in session state
                    st.session_state.documents = all_documents
                    st.session_state.doc_ids = all_doc_ids
                    st.session_state.doc_metadata = all_doc_metadata

                    # Fit TF-IDF
                    if all_documents:
                        vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(all_documents)
                        st.session_state.vectorizer = vectorizer
                        st.session_state.tfidf_matrix = tfidf_matrix
                        
                        # Create chunks for better retrieval using current settings
                        chunk_size = st.session_state.get('chunk_size', 800)
                        overlap = st.session_state.get('overlap', 150)
                        summary_sentences = st.session_state.get('summary_sentences', 3)
                        use_summaries = st.session_state.get('use_summaries', True)

                        chunks, doc_id_mapping = create_chunked_documents(
                            all_documents, all_doc_ids, all_doc_metadata,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            summary_sentences=summary_sentences
                        )
                        st.session_state.chunks = chunks
                        st.session_state.doc_id_mapping = doc_id_mapping
                        
                        # Use summaries or full text for chunk indexing
                        chunk_texts = [c['summary'] if use_summaries else c['text'] for c in chunks]
                        chunk_vectorizer = TfidfVectorizer(stop_words='english')
                        chunk_tfidf_matrix = chunk_vectorizer.fit_transform(chunk_texts)
                        st.session_state.chunk_vectorizer = chunk_vectorizer
                        st.session_state.chunk_tfidf_matrix = chunk_tfidf_matrix
                        # Notify user about auto-rebuild
                        st.success(f"Auto-rebuilt index with {len(chunks)} chunks")

                    message = f"Loaded {new_count} new documents from Zotero (Total: {len(all_documents)} documents)"
                    if duplicates:
                        message += f". Skipped {len(duplicates)} duplicate(s)."
                    st.success(message)
                    if 'chunks' in st.session_state:
                        st.info(f"Index contains {len(st.session_state.chunks)} chunks; using summaries: {st.session_state.get('use_summaries', True)}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading Zotero library: {str(e)}")

# Display persistent load report if it exists
if "last_load_report" in st.session_state:
    report = st.session_state.last_load_report
    
    st.success(f"Last load: {report['new_count']} new documents added from {report['total_processed']} total items")
    
    if report['duplicates'] > 0:
        st.info(f"Skipped {report['duplicates']} duplicates (already in database)")
    
    skipped = report['skipped_items']
    
    # Display detailed skip report
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

    if skipped['errors']:
        st.error(f"Encountered {len(skipped['errors'])} errors:")
        with st.expander("View errors"):
            for item in skipped['errors']:
                st.write(f"- {item}")
    
    # Button to clear the report
    if st.button("Clear Load Report"):
        del st.session_state.last_load_report
        st.rerun()

# Chat interface
st.header("Query the Agile Biofoundry Knowledge Base")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If there's a pending prompt being processed, handle it here (so the UI shows a typing placeholder immediately after submission)
if st.session_state.get('processing') and st.session_state.get('pending_prompt'):
    pending = st.session_state.get('pending_prompt')
    try:
        with st.spinner("Assistant is typing..."):
            # Ensure index exists
            if not ensure_chunk_index():
                result = "No indexed documents available to answer the query. Please load the Zotero library."
            else:
                k_chunks = st.session_state.get('k_chunks', 6)
                similarity_threshold = 0.05

                relevant_chunks, seen_docs = retrieve_relevant_chunks(
                    pending,
                    st.session_state.chunk_vectorizer,
                    st.session_state.chunk_tfidf_matrix,
                    st.session_state.chunks,
                    st.session_state.doc_id_mapping,
                    k=k_chunks,
                    similarity_threshold=similarity_threshold
                )

                use_summaries = st.session_state.get('use_summaries', True)
                context, cited_docs = format_context_from_chunks(relevant_chunks, seen_docs, use_summaries=use_summaries)

                if not context or context == "":
                    context = "No relevant documents found."

                result = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant knowledgeable about Agile Biofoundry. Use the provided context to answer the query comprehensively."},
                        {"role": "user", "content": f"Context from knowledge base:\n{context}\n\nQuery: {pending}"}
                    ]
                ).choices[0].message.content

                if cited_docs:
                    result += "\n\n---\n**Sources:**\n"
                    for doc in cited_docs:
                        result += f"- {doc['title']} (ID: {doc['id']}, Relevance: {doc['similarity']:.2%}, Chunks: {doc['chunk_count']})\n"
    except Exception as e:
        result = f"Error generating response: {str(e)}"

    # Replace the placeholder assistant message with the actual result
    assistant_idx = st.session_state.get('pending_assistant_index')
    if assistant_idx is not None and assistant_idx < len(st.session_state.messages):
        st.session_state.messages[assistant_idx]['content'] = result
    else:
        st.session_state.messages.append({"role": "assistant", "content": result})

    # Clear pending flags and stop processing
    st.session_state.pending_prompt = None
    st.session_state.pending_assistant_index = None
    st.session_state.processing = False

    st.rerun()

# User input
if prompt := st.chat_input("Ask a question about Agile Biofoundry:"):
    # Append user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add assistant placeholder and mark processing, then rerun so the UI shows placeholder immediately
    st.session_state.messages.append({"role": "assistant", "content": "...typing..."})
    st.session_state.pending_prompt = prompt
    st.session_state.pending_assistant_index = len(st.session_state.messages) - 1
    st.session_state.processing = True
    st.rerun()