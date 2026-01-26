"""
Article Storage Module
Handles persistent storage of Zotero articles and prevents duplicates
"""

import json
import os
from typing import List, Dict, Tuple

ARTICLES_FILE = "zotero_articles.json"


def load_articles() -> Dict:
    """Load articles from persistent storage"""
    if os.path.exists(ARTICLES_FILE):
        try:
            with open(ARTICLES_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"articles": [], "metadata": {}}
    return {"articles": [], "metadata": {}}


def save_articles(data: Dict) -> None:
    """Save articles to persistent storage"""
    with open(ARTICLES_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def article_exists(zotero_id: str, articles_data: Dict) -> bool:
    """Check if an article already exists by Zotero ID"""
    return zotero_id in articles_data["metadata"]


def add_article(
    zotero_id: str,
    content: str,
    title: str,
    item_type: str,
    abstract: str = ""
) -> Tuple[bool, str]:
    """
    Add an article to storage if it doesn't already exist
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    articles_data = load_articles()
    
    if article_exists(zotero_id, articles_data):
        return False, f"Article '{title}' already exists (ID: {zotero_id})"
    
    # Add article content
    articles_data["articles"].append(content)
    
    # Store metadata with Zotero ID as key
    articles_data["metadata"][zotero_id] = {
        "title": title,
        "itemType": item_type,
        "abstract": abstract[:200] if abstract else "",
        "index": len(articles_data["articles"]) - 1
    }
    
    save_articles(articles_data)
    return True, f"Article '{title}' added successfully"


def get_all_articles() -> Tuple[List[str], List[str], List[Dict]]:
    """
    Get all stored articles
    
    Returns:
        Tuple of (documents, doc_ids, doc_metadata)
    """
    articles_data = load_articles()
    documents = articles_data["articles"]
    
    # Reconstruct doc_ids and doc_metadata in correct order
    doc_ids = []
    doc_metadata = []
    
    for zotero_id, metadata in articles_data["metadata"].items():
        index = metadata.get("index")
        if index is not None and index < len(documents):
            doc_ids.append(zotero_id)
            doc_metadata.append({
                "title": metadata.get("title", "Untitled"),
                "itemType": metadata.get("itemType", "Unknown"),
                "abstract": metadata.get("abstract", "")
            })
    
    return documents, doc_ids, doc_metadata


def remove_article(zotero_id: str) -> Tuple[bool, str]:
    """Remove an article from storage"""
    articles_data = load_articles()
    
    if not article_exists(zotero_id, articles_data):
        return False, "Article not found"
    
    metadata = articles_data["metadata"][zotero_id]
    index = metadata.get("index")
    
    # Remove from articles list
    if index is not None and index < len(articles_data["articles"]):
        articles_data["articles"].pop(index)
        
        # Update indices in metadata
        for zid, meta in articles_data["metadata"].items():
            if meta.get("index", -1) > index:
                meta["index"] -= 1
    
    # Remove metadata
    del articles_data["metadata"][zotero_id]
    
    save_articles(articles_data)
    title = metadata.get("title", "Untitled")
    return True, f"Article '{title}' removed"


def clear_all_articles() -> str:
    """Clear all stored articles"""
    save_articles({"articles": [], "metadata": {}})
    return "All articles cleared"


def get_article_count() -> int:
    """Get count of stored articles"""
    articles_data = load_articles()
    return len(articles_data["articles"])
