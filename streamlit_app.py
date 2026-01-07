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
    st.session_state.documents = []
if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = []  # Store titles and other metadata

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
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
                
                # Delete button
                if st.button(f"Delete", key=f"delete_{idx}"):
                    # Remove from all lists
                    del st.session_state.documents[idx]
                    del st.session_state.doc_ids[idx]
                    del st.session_state.doc_metadata[idx]
                    
                    # Refit TF-IDF if documents remain
                    if st.session_state.documents:
                        vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(st.session_state.documents)
                        st.session_state.vectorizer = vectorizer
                        st.session_state.tfidf_matrix = tfidf_matrix
                    else:
                        # Clear vectorizer if no documents
                        if 'vectorizer' in st.session_state:
                            del st.session_state.vectorizer
                        if 'tfidf_matrix' in st.session_state:
                            del st.session_state.tfidf_matrix
                    
                    st.rerun()
        
        # Bulk actions
        st.subheader("Bulk Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear All", type="secondary"):
                st.session_state.documents = []
                st.session_state.doc_ids = []
                st.session_state.doc_metadata = []
                if 'vectorizer' in st.session_state:
                    del st.session_state.vectorizer
                if 'tfidf_matrix' in st.session_state:
                    del st.session_state.tfidf_matrix
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

                progress_bar = st.progress(0)
                total_items = len(items)

                for item_idx, item in enumerate(items):
                    # Update progress
                    progress_bar.progress((item_idx + 1) / total_items)
                    
                    # Skip if not a relevant item type
                    if item['data']['itemType'] not in ['journalArticle', 'webpage', 'report', 'conferencePaper']:
                        continue

                    # Extract metadata text
                    title = item['data'].get('title', 'Untitled')
                    abstract = item['data'].get('abstractNote', '')
                    item_type = item['data']['itemType']
                    notes = []

                    # Get child notes
                    children = zot.children(item['key'])
                    for child in children:
                        if child['data']['itemType'] == 'note':
                            notes.append(child['data'].get('note', ''))

                    text = f"{title}\n{abstract}\n{' '.join(notes)}"

                    # If there are attachments (e.g., PDF or snapshot)
                    for child in children:
                        if child['data']['itemType'] == 'attachment':
                            link_mode = child['data'].get('linkMode', '')
                            if link_mode in ['linked_file', 'imported_file', 'imported_url']:
                                # Get the file URL via API
                                file_url = f"https://api.zotero.org/{zotero_library_type}s/{zotero_library_id}/items/{child['key']}/file?key={zotero_api_key}"
                                response = requests.get(file_url)
                                if response.status_code == 200:
                                    content_type = response.headers.get('Content-Type', '')
                                    if 'application/pdf' in content_type:
                                        pdf_text = extract_pdf_text(response.content)
                                        text += f"\n{pdf_text}"
                                    elif 'text/html' in content_type:
                                        text += f"\n{response.text}"

                    if text.strip():
                        documents.append(text)
                        doc_ids.append(item['key'])
                        doc_metadata.append({
                            'title': title,
                            'itemType': item_type,
                            'abstract': abstract[:200] if abstract else ''
                        })

                progress_bar.empty()

                if not documents:
                    st.error("No documents found in the library.")
                else:
                    # Store documents in session state
                    st.session_state.documents = documents
                    st.session_state.doc_ids = doc_ids
                    st.session_state.doc_metadata = doc_metadata

                    # Fit TF-IDF
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(documents)
                    st.session_state.vectorizer = vectorizer
                    st.session_state.tfidf_matrix = tfidf_matrix

                    st.success(f"Loaded {len(documents)} documents from Zotero.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading Zotero library: {str(e)}")

# Chat interface
st.header("Query the Agile Biofoundry Knowledge Base")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about Agile Biofoundry:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "documents" not in st.session_state or not st.session_state.documents or not openai_api_key:
        response = "Please load the Zotero library and ensure OpenAI API key is set in secrets."
    else:
        try:
            # Retrieve relevant documents using TF-IDF
            vectorizer = st.session_state.vectorizer
            tfidf_matrix = st.session_state.tfidf_matrix
            query_vec = vectorizer.transform([prompt])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3 documents
            context = ""
            cited_docs = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold for relevance
                    context += f"\n\nDocument ID: {st.session_state.doc_ids[idx]}\n{st.session_state.documents[idx][:1000]}..."
                    cited_docs.append({
                        'title': st.session_state.doc_metadata[idx].get('title', 'Untitled'),
                        'id': st.session_state.doc_ids[idx],
                        'similarity': float(similarities[idx])
                    })

            if not context:
                context = "No relevant documents found."

            # Prompt OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant knowledgeable about Agile Biofoundry. Use the provided context to answer the query."},
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {prompt}"}
                ]
            ).choices[0].message.content
            
            # Add citations if documents were used
            if cited_docs:
                response += "\n\n---\n**Sources:**\n"
                for doc in cited_docs:
                    response += f"- {doc['title']} (ID: {doc['id']}, Relevance: {doc['similarity']:.2%})\n"

        except Exception as e:
            response = f"Error generating response: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)