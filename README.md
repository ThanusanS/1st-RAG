# Document AI RAG

This is a small Retrieval-Augmented Generation app:

This project is a production-style local RAG MVP:

1. Uploads PDF, Markdown, and text files.
2. Extracts page-aware text and chunks it with overlap.
3. Creates Gemini embeddings and persists them in ChromaDB.
4. Retrieves semantic matches and generates cited answers with Gemini.
5. Exposes both a Streamlit UI and FastAPI endpoints.

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

Run the UI:

```powershell
streamlit run app.py
```

Run the API:

```powershell
uvicorn api:app --reload
```

The API provides `/health`, `/documents`, and `/ask`. Set `APP_USERNAME` and
`APP_PASSWORD` in `.env` to enable the Streamlit login.
