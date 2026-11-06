from pathlib import Path
import os
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def read_documents() -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    DATA_DIR.mkdir(exist_ok=True)

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue

        if text.strip():
            documents.append({"source": path.name, "text": text.strip()})

    return documents


def split_into_chunks(documents: list[dict[str, str]], chunk_size: int = 900) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for document in documents:
        words = document["text"].split()
        for start in range(0, len(words), chunk_size):
            text = " ".join(words[start : start + chunk_size]).strip()
            if text:
                chunks.append({"source": document["source"], "text": text})
    return chunks


def retrieve(question: str, chunks: list[dict[str, str]], count: int = 4) -> list[dict[str, str]]:
    if not chunks:
        return []

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        stop_words=None,
    )
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix)[0]
    best_indexes = scores.argsort()[::-1][:count]

    return [
        {**chunks[index], "score": f"{scores[index]:.3f}"}
        for index in best_indexes
        if scores[index] > 0
    ]


def answer_question(question: str, context: list[dict[str, str]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

    context_text = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}" for item in context
    )
    prompt = f"""You answer questions using only the supplied context.
If the answer is not in the context, say you do not know.
Be concise and cite sources in square brackets, for example [guide.md].

Context:
{context_text}

Question: {question}
"""

    client = genai.Client(api_key=api_key)
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "12"))
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            break
        except errors.ServerError as error:
            if error.code != 503 or attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 30))
    return response.text or "Gemini returned an empty answer."


st.set_page_config(page_title="My first RAG app", page_icon="📚")
st.title("📚 My first RAG app")
st.caption("Ask questions about the files in the data folder.")

documents = read_documents()
chunks = split_into_chunks(documents)

with st.sidebar:
    st.header("Knowledge base")
    st.write(f"{len(documents)} document(s), {len(chunks)} chunk(s)")
    st.info("Add .txt, .md, or .pdf files to the data folder, then refresh this page.")
    st.code(f"Model: {MODEL}")

question = st.text_input("Ask a question")
if question:
    if not chunks:
        st.warning("Add at least one document to the data folder first.")
    else:
        with st.spinner("Searching documents and asking Gemini..."):
            matches = retrieve(question, chunks)
            if not matches:
                st.warning("I could not find relevant content in your documents.")
            else:
                try:
                    st.subheader("Answer")
                    st.write(answer_question(question, matches))
                    with st.expander("Retrieved sources"):
                        for item in matches:
                            st.markdown(f"**{item['source']}** (score: {item['score']})")
                            st.write(item["text"])
                except RuntimeError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Gemini request failed: {error}")
