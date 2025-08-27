from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.rag_engine import get_answer

app = FastAPI()

# 🔓 Enable CORS so frontend can access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://localhost:3000"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Input schema
class Query(BaseModel):
    question: str

# 🔍 Endpoint to query the RAG system
@app.post("/query")
async def query_rag(data: Query):
    answer = get_answer(data.question)
    return {"answer": answer}
