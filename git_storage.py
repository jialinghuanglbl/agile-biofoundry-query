"""
Git-based persistent storage for Streamlit Cloud
Automatically commits article changes to the repository
"""

import os
import json
import subprocess
from typing import Tuple

def _get_repo_root() -> str:
    """Get the repository root directory"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    # Fallback: assume we're in the repo root
    return os.path.dirname(os.path.abspath(__file__))


def _get_data_dir() -> str:
    """Get the data directory for storing articles"""
    repo_root = _get_repo_root()
    data_dir = os.path.join(repo_root, "zotero_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _git_configured() -> bool:
    """Check if git is configured and we're in a repo"""
    try:
        subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        return True
    except Exception:
        return False


def auto_commit_changes(collection_name: str, message: str) -> bool:
    """
    Automatically commit changes to git (for Streamlit Cloud persistence)
    
    Args:
        collection_name: Name of the collection that changed
        message: Commit message
    
    Returns:
        True if commit succeeded, False otherwise
    """
    if not _git_configured():
        # Git not configured - silently fail, fall back to file storage only
        return False
    
    try:
        repo_root = _get_repo_root()
        data_dir = _get_data_dir()
        
        # Stage the article file
        article_file = os.path.join(data_dir, f"zotero_articles_{collection_name}.json")
        if os.path.exists(article_file):
            subprocess.run(
                ["git", "add", article_file],
                cwd=repo_root,
                capture_output=True,
                timeout=10,
                check=True
            )
        
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain", "zotero_data/"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout.strip():
            # There are changes; commit them
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_root,
                capture_output=True,
                timeout=10,
                check=True
            )
            
            # Try to push (may fail if no upstream, but that's OK)
            try:
                subprocess.run(
                    ["git", "push"],
                    cwd=repo_root,
                    capture_output=True,
                    timeout=15
                )
            except Exception:
                pass  # Push failed, but commit succeeded
            
            return True
        
        return False
    
    except Exception as e:
        # Git operations failed; fall back to file-only storage
        return False


def ensure_data_persisted(collection_name: str) -> str:
    """
    Ensure data is persisted, attempting git commit if on Streamlit Cloud
    
    Returns:
        Status message
    """
    data_dir = _get_data_dir()
    article_file = os.path.join(data_dir, f"zotero_articles_{collection_name}.json")
    
    if not os.path.exists(article_file):
        return "No articles to persist"
    
    if auto_commit_changes(collection_name, f"Auto-commit: Update {collection_name} articles"):
        return "Committed to git"
    else:
        return "Stored locally only (git unavailable)"


def pull_latest_articles() -> bool:
    """
    Pull the latest article data from git (for Streamlit Cloud startup)
    
    Returns:
        True if pull succeeded, False otherwise
    """
    if not _git_configured():
        return False
    
    try:
        repo_root = _get_repo_root()
        
        # Pull the latest changes
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        return result.returncode == 0
    
    except Exception as e:
        return False
