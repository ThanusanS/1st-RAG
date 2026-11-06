from types import SimpleNamespace

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag_core import RAGError, RAGService

app = FastAPI(title="Document AI API", version="1.0.0")
service = RAGService.from_environment()


class Question(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, int | str]:
    return {"status": "ok", "chunks": service.count()}


@app.post("/documents")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, int]:
    try:
        uploads = []
        for file in files:
            content = await file.read()
            uploads.append(
                SimpleNamespace(name=file.filename or "upload", getvalue=lambda content=content: content)
            )
        return {"indexed_chunks": service.index_uploads(uploads)}
    except RAGError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/ask")
def ask(question: Question) -> dict[str, object]:
    try:
        answer, sources = service.ask(question.question)
        return {"answer": answer, "sources": sources}
    except RAGError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
