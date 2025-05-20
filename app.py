# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import chromadb
import hashlib
from sentence_transformers import SentenceTransformer

app = FastAPI()
chroma_client = chromadb.PersistentClient(path="chroma_db")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_path_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()


class QueryRequest(BaseModel):
    path: str
    query: str


@app.post("/ask")
async def ask_code(request: QueryRequest):
    path_hash = get_path_hash(request.path)
    collection_name = "project_index_" + path_hash[:10]

    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        return {"error": f"No index found for path: {request.path}" +
                f"Existing collections: {chroma_client.list_collections()}"}

    # Embed query and search
    query_embedding = embedding_model.encode([request.query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    context_chunks = results["documents"][0]
    context = "\n---\n".join(context_chunks)

    prompt = f"""You are an assistant that answers questions about code.
Context:
{context}

Question: {request.query}
Answer:"""

    response = ollama.chat(model="deepseek-coder:6.7b", messages=[
        {"role": "user", "content": prompt}
    ])

    return {"response": response["message"]["content"]}
