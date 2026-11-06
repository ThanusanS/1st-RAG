# My first RAG project

This is a small Retrieval-Augmented Generation app:

1. It reads `.txt`, `.md`, and `.pdf` files from `data/`.
2. It retrieves the most relevant chunks using local TF-IDF search.
3. It sends only those chunks to Gemini 2.5 Flash.
4. It displays an answer with source names.

## Setup on Windows

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your Google AI Studio API key.
Do not share the key or commit `.env`.

Add your own files to `data/`, then run:

```powershell
streamlit run app.py
```

If `gemini-3.8-flash` is unavailable for your account, change `GEMINI_MODEL`
in `.env` to a model available in Google AI Studio.
