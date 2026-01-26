# agile-biofoundry-query
Using Zotero
## Features

### Article Storage
- Articles from Zotero are automatically saved to persistent storage (`zotero_articles.json`)
- Duplicate articles are automatically detected and prevented - when loading from Zotero, articles that have already been imported are skipped
- Articles remain available even after restarting the application
- Users can delete individual articles or clear all articles through the UI
- All changes are persisted to the storage file

### Smart Document Retrieval
- **Document Chunking**: Long articles are split into meaningful chunks (1500 characters with 300-character overlap) to preserve context while enabling fine-grained retrieval
- **Enhanced Context**: The chatbot retrieves up to 8 relevant chunks instead of just 1000 characters from 3 documents
- **Multi-chunk Support**: When multiple relevant chunks are found from the same document, they're all included in the context
- **Better Answers**: More comprehensive article content enables the AI to provide more detailed and accurate responses

### Article Storage Module (`article_storage.py`)
The `article_storage` module provides functions to manage article persistence:

- `load_articles()` - Load articles from persistent storage
- `save_articles()` - Save articles to JSON file
- `article_exists(zotero_id)` - Check if an article is already stored
- `add_article(zotero_id, content, title, item_type, abstract)` - Add new article with duplicate check
- `get_all_articles()` - Retrieve all stored articles
- `remove_article(zotero_id)` - Delete a specific article
- `clear_all_articles()` - Delete all articles
- `get_article_count()` - Get total number of articles

### Document Retrieval Module (`document_retrieval.py`)
The `document_retrieval` module implements intelligent document chunking and retrieval:

- `chunk_document(document, chunk_size, overlap)` - Split documents into overlapping chunks
- `create_chunked_documents(documents, doc_ids, doc_metadata)` - Create chunks for all documents
- `retrieve_relevant_chunks(...)` - Find the most relevant chunks using TF-IDF scoring
- `format_context_from_chunks(...)` - Format chunks into readable context for the LLM

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Set up Zotero credentials in `.streamlit/secrets.toml`
3. Run the app: `streamlit run streamlit_app.py`
4. Click "Load Zotero Library" to import articles
5. Articles are automatically saved, chunked, and indexed
6. Ask questions in the chat interface - the bot will retrieve relevant content from multiple sections