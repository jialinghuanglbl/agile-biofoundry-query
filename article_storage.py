"""
Article Storage Module
Supports multiple collections with separate storage files

Articles are stored locally in ./zotero_data/
"""

import json
import os
from typing import List, Dict, Tuple


def _get_data_dir():
    """Return the local data directory for storing articles."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "zotero_data")


def _get_articles_file(collection_name: str = "default") -> str:
    """Get the file path for a specific collection."""
    data_dir = _get_data_dir()
    return os.path.join(data_dir, f"zotero_articles_{collection_name}.json")


def load_articles(collection_name: str = "default") -> Dict:
    """Load articles from persistent storage for a specific collection"""
    articles_file = _get_articles_file(collection_name)
    if os.path.exists(articles_file):
        try:
            with open(articles_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"articles": [], "metadata": {}}
    return {"articles": [], "metadata": {}}


def save_articles(data: Dict, collection_name: str = "default") -> None:
    """Save articles to persistent storage using an atomic write."""
    articles_file = _get_articles_file(collection_name)
    os.makedirs(os.path.dirname(articles_file), exist_ok=True)
    tmp_path = articles_file + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, articles_file)


def article_exists(zotero_id: str, articles_data: Dict = None, collection_name: str = "default") -> bool:
    """Check if an article already exists by Zotero ID"""
    if articles_data is None:
        articles_data = load_articles(collection_name)
    return zotero_id in articles_data.get("metadata", {})


def add_article(
    zotero_id: str,
    content: str,
    title: str,
    item_type: str,
    abstract: str = "",
    collection_name: str = "default",
    low_relevance: bool = False,
    url: str = ""
) -> Tuple[bool, str]:
    """Add an article to storage if it doesn't already exist"""
    articles_data = load_articles(collection_name)
    
    if article_exists(zotero_id, articles_data, collection_name):
        return False, f"Article '{title}' already exists (ID: {zotero_id})"
    
    articles_data.setdefault("articles", []).append(content)
    articles_data.setdefault("metadata", {})[zotero_id] = {
        "title": title,
        "itemType": item_type,
        "abstract": abstract[:200] if abstract else "",
        "index": len(articles_data["articles"]) - 1,
        "low_relevance": low_relevance,
        "url": url or ""
    }
    
    save_articles(articles_data, collection_name)
    return True, f"Article '{title}' added successfully"


def get_all_articles(collection_name: str = "default") -> Tuple[List[str], List[str], List[Dict]]:
    """Get all stored articles for a collection"""
    articles_data = load_articles(collection_name)
    documents = articles_data.get("articles", [])
    
    metadata = articles_data.get("metadata", {})
    sorted_items = sorted(metadata.items(), key=lambda kv: kv[1].get("index", 0))

    doc_ids = []
    doc_metadata = []

    for zotero_id, meta in sorted_items:
        index = meta.get("index")
        if index is not None and index < len(documents):
            doc_ids.append(zotero_id)
            doc_metadata.append({
                "title": meta.get("title", "Untitled"),
                "itemType": meta.get("itemType", "Unknown"),
                "abstract": meta.get("abstract", ""),
                "low_relevance": meta.get("low_relevance", False),
                "url": meta.get("url", "")
            })

    return documents, doc_ids, doc_metadata


def remove_article(zotero_id: str, collection_name: str = "default") -> Tuple[bool, str]:
    """Remove an article from storage"""
    articles_data = load_articles(collection_name)
    
    if not article_exists(zotero_id, articles_data, collection_name):
        return False, "Article not found"
    
    metadata = articles_data["metadata"][zotero_id]
    index = metadata.get("index")
    
    if index is not None and index < len(articles_data["articles"]):
        articles_data["articles"].pop(index)
        for zid, meta in articles_data["metadata"].items():
            if meta.get("index", -1) > index:
                meta["index"] -= 1
    
    del articles_data["metadata"][zotero_id]
    save_articles(articles_data, collection_name)
    title = metadata.get("title", "Untitled")
    return True, f"Article '{title}' removed"


def rename_article(zotero_id: str, new_title: str, collection_name: str = "default") -> Tuple[bool, str]:
    """Rename an article's title in storage"""
    articles_data = load_articles(collection_name)
    
    if not article_exists(zotero_id, articles_data, collection_name):
        return False, "Article not found"
    
    articles_data["metadata"][zotero_id]["title"] = new_title
    save_articles(articles_data, collection_name)
    return True, f"Article '{zotero_id}' renamed to '{new_title}'"


def clear_all_articles(collection_name: str = "default") -> str:
    """Clear all stored articles for a collection"""
    save_articles({"articles": [], "metadata": {}}, collection_name)
    return "All articles cleared"


def get_article_count(collection_name: str = "default") -> int:
    """Get count of stored articles for a collection"""
    articles_data = load_articles(collection_name)
    return len(articles_data.get("articles", []))