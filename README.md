# agile-biofoundry-query
Using Zotero
## Features

### Article Storage
- Articles from Zotero are automatically saved to persistent storage (`zotero_articles.json`)
- Duplicate articles are automatically detected and prevented - when loading from Zotero, articles that have already been imported are skipped
- Articles remain available even after restarting the application
- Users can delete individual articles or clear all articles through the UI
- All changes are persisted to the storage file

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

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Set up Zotero credentials in `.streamlit/secrets.toml`
3. Run the app: `streamlit run streamlit_app.py`
4. Click "Load Zotero Library" to import articles
5. Articles are automatically saved and duplicates are prevented