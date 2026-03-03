# agile-biofoundry-query

This Streamlit app lets you import items from Zotero collections and query them using OpenAI-powered chat.  It handles PDFs, transcripts, and more, chunking documents for efficient retrieval.

**New:** when responding to queries, the assistant now scans chunks for page numbers and timestamp patterns and includes these details in source citations, helping you trace answers back to specific parts of a document.

## Persistence options
Because Streamlit Cloud’s filesystem is ephemeral, saved articles can disappear if the container is restarted. To make storage durable you can
enable git-backed persistence:

1. Add your GitHub Personal Access Token to Streamlit secrets as `github_token`.
2. Add a secret `enable_git_persistence` with value `true`.

With those in place the app will commit and push `zotero_data/*.json` every time articles are added or removed, and will pull the latest data on startup. You can also press **Sync articles to GitHub** in the sidebar to force a push.

Alternatively, keep the existing uptime/monitor approach for short-term waking but be aware it does not guarantee data persistence.
