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


def _configure_git_if_needed() -> bool:
    """
    Configure git user if not already configured.
    Returns True if git is configured/was successfully configured.
    """
    try:
        # Check if user.email is already configured
        result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return True  # Already configured
        
        # Not configured; set up with default values
        # Use environment variables if available, otherwise use defaults
        git_email = os.environ.get("GIT_EMAIL", "streamlit-bot@localhost")
        git_name = os.environ.get("GIT_NAME", "Streamlit Bot")
        
        subprocess.run(
            ["git", "config", "user.email", git_email],
            capture_output=True,
            timeout=5
        )
        subprocess.run(
            ["git", "config", "user.name", git_name],
            capture_output=True,
            timeout=5
        )
        
        return True
    except Exception as e:
        return False


def _ensure_github_token() -> bool:
    """
    Ensure GitHub token is configured for pushing.
    Tries multiple sources: environment, Streamlit secrets, .netrc
    """
    try:
        # Try multiple sources for the token (in priority order)
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("github_token")
        
        # Fallback: try Streamlit secrets if token not in environment
        if not github_token:
            try:
                # Only import if needed, to avoid issues in non-Streamlit contexts
                import sys
                if 'streamlit' in sys.modules:
                    import streamlit as st
                    try:
                        github_token = st.secrets.get("github_token")
                    except Exception:
                        pass
            except Exception:
                pass
        
        if not github_token:
            return False  # No token available
        
        # Store in environment for subprocess calls
        os.environ["GITHUB_TOKEN"] = github_token
        
        # Create .netrc file for git authentication
        try:
            netrc_path = os.path.expanduser("~/.netrc")
            
            has_github = False
            if os.path.exists(netrc_path):
                try:
                    with open(netrc_path, 'r') as f:
                        content = f.read()
                        has_github = 'github.com' in content
                except Exception:
                    pass
            
            if not has_github:
                # Append github creds to .netrc
                with open(netrc_path, 'a' if os.path.exists(netrc_path) else 'w') as f:
                    f.write(f"\nmachine github.com\nlogin git\npassword {github_token}\n")
                os.chmod(netrc_path, 0o600)
        except Exception:
            # .netrc creation failed; continue with other methods
            pass
        
        return True
    except Exception:
        return False


def _git_configured() -> bool:
    """Check if git is configured and we're in a repo"""
    try:
        _configure_git_if_needed()  # Auto-configure if needed
        _ensure_github_token()  # Set up token if available
        
        result = subprocess.run(
            ["git", "status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def auto_commit_changes(collection_name: str, message: str) -> bool:
    """
    Automatically commit changes to git (for Streamlit Cloud persistence)
    Commits locally and PUSHES to GitHub immediately.
    
    Args:
        collection_name: Name of the collection that changed
        message: Commit message
    
    Returns:
        True if both commit AND push succeeded, False otherwise
    """
    if not _git_configured():
        # Git not configured - silently fail, fall back to file storage only
        return False
    
    try:
        repo_root = _get_repo_root()
        data_dir = _get_data_dir()
        
        # Set up environment with GitHub token and auth settings
        env = os.environ.copy()
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("github_token")
        if github_token:
            # These help git authenticate without prompting
            env['GIT_ASKPASS_ARGS'] = github_token
            env['GIT_TERMINAL_PROMPT'] = '0'  # Don't prompt for input
        
        # Stage the article file
        article_file = os.path.join(data_dir, f"zotero_articles_{collection_name}.json")
        if os.path.exists(article_file):
            subprocess.run(
                ["git", "add", article_file],
                cwd=repo_root,
                capture_output=True,
                timeout=10,
                check=True,
                env=env
            )
        
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain", "zotero_data/"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            env=env
        )
        
        if not result.stdout.strip():
            # No changes to commit
            return False
        
        # Commit the changes
        try:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_root,
                capture_output=True,
                timeout=10,
                check=True,
                env=env
            )
        except subprocess.CalledProcessError:
            # Commit failed (maybe nothing staged)
            return False
        
        # NOW: PUSH to GitHub immediately
        # This is critical for Streamlit Cloud persistence
        push_succeeded = False
        push_error = ""
        
        for branch in ["main", "master"]:
            try:
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", branch],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    env=env
                )
                
                if push_result.returncode == 0:
                    push_succeeded = True
                    break
                else:
                    # Push failed; log the error for debugging
                    push_error = push_result.stderr or push_result.stdout
                    # Log to help diagnose issues with GitHub authentication
                    import sys
                    try:
                        print(f"[git_storage] Push to {branch} failed: {push_error[:200]}", file=sys.stderr)
                    except Exception:
                        pass
                    
                    if "Authentication failed" in push_error or "401" in push_error:
                        # Auth failed; token might be wrong/expired
                        continue
            except subprocess.TimeoutExpired:
                # Push timed out; try next branch
                continue
            except Exception:
                # Other error; try next branch
                continue
        
        return push_succeeded
    
    except Exception as e:
        # Git operations failed completely
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
        
        # Set up environment with GitHub token
        env = os.environ.copy()
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("github_token")
        if github_token:
            env['GIT_AUTH_HELPER'] = 'netrc'
        
        # First, ensure we're on the right branch
        for branch in ["main", "master"]:
            try:
                subprocess.run(
                    ["git", "checkout", branch],
                    cwd=repo_root,
                    capture_output=True,
                    timeout=10,
                    env=env
                )
                break
            except Exception:
                continue
        
        # Fetch to get latest from remote
        try:
            fetch_result = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=15,
                env=env
            )
            if fetch_result.returncode != 0:
                import sys
                try:
                    print(f"[git_storage] Fetch failed: {fetch_result.stderr[:200]}", file=sys.stderr)
                except Exception:
                    pass
        except Exception as e:
            import sys
            try:
                print(f"[git_storage] Fetch error: {str(e)[:200]}", file=sys.stderr)
            except Exception:
                pass
        
        # Try to pull the latest changes
        for branch in ["main", "master"]:
            result = subprocess.run(
                ["git", "pull", "--ff-only", "origin", branch],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=15,
                env=env
            )
            
            if result.returncode == 0:
                return True
        
        return False
    
    except Exception as e:
        return False
