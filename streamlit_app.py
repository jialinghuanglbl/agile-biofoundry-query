import streamlit as st
from pyzotero import zotero
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=openai_api_key)
import PyPDF2
import requests
import io
import os

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


# Button to load library
if st.button("Load Zotero Library"):
    if not zotero_library_id or not zotero_api_key:
        st.error("Zotero Library ID and API Key must be set in Streamlit secrets.")
    else:
        try:
            zot = zotero.Zotero(zotero_library_id, zotero_library_type, zotero_api_key)

            # Fetch items from collection if key provided, else all items
            if zotero_collection_key:
                items = zot.collection_items(zotero_collection_key)
            else:
                items = zot.items()

            documents = []
            doc_ids = []

            for item in items:
                # Skip if not a relevant item type
                # Fix: itemType is in item['data'], not item['meta']
                if item['data']['itemType'] not in ['journalArticle', 'webpage', 'report', 'conferencePaper']:
                    continue

                # Extract metadata text
                # Fix: These are also in item['data']
                title = item['data'].get('title', '')
                abstract = item['data'].get('abstractNote', '')
                notes = []

                # Get child notes
                children = zot.children(item['key'])
                for child in children:
                    # Fix: Check itemType in child['data']
                    if child['data']['itemType'] == 'note':
                        notes.append(child['data'].get('note', ''))

                text = f"{title}\n{abstract}\n{' '.join(notes)}"

                # If there are attachments (e.g., PDF or snapshot)
                for child in children:
                    if child['data']['itemType'] == 'attachment':
                        # Fix: linkMode is also in child['data']
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
                                elif 'text/html' in content_type:  # For web snapshots
                                    text += f"\n{response.text}"

                if text.strip():
                    documents.append(text)
                    doc_ids.append(item['key'])

            if not documents:
                st.error("No documents found in the library.")
            else:
                # Store documents in session state
                st.session_state.documents = documents
                st.session_state.doc_ids = doc_ids

                # Fit TF-IDF
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                st.session_state.vectorizer = vectorizer
                st.session_state.tfidf_matrix = tfidf_matrix

                st.success(f"Loaded {len(documents)} documents from Zotero.")
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

    if "documents" not in st.session_state or not openai_api_key:
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
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold for relevance
                    context += f"\n\nDocument ID: {st.session_state.doc_ids[idx]}\n{st.session_state.documents[idx][:1000]}..."  # Truncate to 1000 chars

            if not context:
                context = "No relevant documents found."

            # Prompt OpenAI
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant knowledgeable about Agile Biofoundry. Use the provided context to answer the query."},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {prompt}"}
            ]).choices[0].message.content

        except Exception as e:
            response = f"Error generating response: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)