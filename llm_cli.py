# llm_cli.py
import argparse
import httpx
from rich.console import Console
import subprocess
import shutil
import json
import time
import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

console = Console()
chroma_client = chromadb.PersistentClient(path="chroma_db")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_RESULTS = 20
TEMPERATURE = 0.3


def is_ollama_running():
    try:
        httpx.get("http://localhost:11434", timeout=2)
        return True
    except httpx.RequestError:
        return False


def try_start_ollama():
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        console.print(
            "[red]Ollama is not installed. Please install it from https://ollama.com/download[/red]")
        return False

    try:
        subprocess.Popen([ollama_path, "serve"])
        time.sleep(2)
        if is_ollama_running():
            console.print("[green]✅ Ollama started successfully[/green]")
            return True
        else:
            console.print("[red]❌ Failed to start Ollama[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Error starting Ollama: {e}[/red]")
        return False


def ensure_model_available(model: str):
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        tags = response.json().get("models", [])
        local_models = [m["name"] for m in tags]
        if model in local_models:
            return
        console.print(f"[yellow]Model '{
                      model}' not found locally. Pulling...[/yellow]")
        subprocess.run(["ollama", "pull", model], check=True)
        console.print(f"[green]✅ Model '{model}' pulled successfully[/green]")
    except Exception as e:
        console.print(f"[red]Failed to check or pull model: {e}[/red]")
        exit(1)


def get_path_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()


def get_context_from_embeddings(path: str, query: str) -> str:
    path_hash = get_path_hash(path)
    collection_name = "project_index_" + path_hash[:10]

    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        raise RuntimeError(f"No index found for path: {path}")

    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding], n_results=EMBEDDING_RESULTS)
    context_chunks = results["documents"][0]
    return "\n---\n".join(context_chunks)


def build_prompt(context: str, query: str) -> str:
    return f"""You are an assistant that answers questions about code based ONLY on the provided context.
You are permitted to make suggestions if you can't bet your life on it that it's from the context
If the answer is not contained within the context, say "I don't know."
Context:
{context}

Question: {query} in the context
Answer:"""


def stream_local_chat(model: str, prompt: str):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": TEMPERATURE
        }
    }

    with httpx.stream("POST", url, headers=headers, json=payload, timeout=None) as response:
        if response.status_code != 200:
            console.print(
                f"[red]Error: {response.status_code} - {response.text}[/red]")
            return

        for line in response.iter_lines():
            if line.strip():
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    console.print(token, end="", style="cyan")
                except Exception as e:
                    console.print(f"[red]Error parsing response: {e}[/red]")


def stream_remote_chat(path: str, query: str):
    url = "http://localhost:8000/ask"
    payload = {"path": path, "query": query}

    try:
        response = httpx.post(url, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json().get("response")
        console.print(content, style="cyan")
    except Exception as e:
        console.print(f"[red]Failed to query remote API: {e}[/red]")


def is_path_indexed(path: str) -> bool:
    path_hash = get_path_hash(path)
    collection_name = "project_index_" + path_hash[:10]
    try:
        collection = chroma_client.get_collection(name=collection_name)
        ids = collection.get(limit=1).get("ids", [])
        return len(ids) > 0
    except Exception:
        return False


def run_indexer(path: str, force: bool):
    if not force and is_path_indexed(path):
        console.print(f"[green]Index already exists for: {path}[/green]")
        return

    console.print(f"[blue]Indexing code at {path}...[/blue]")
    try:
        subprocess.run(["python", "index_code.py", path], check=True)
    except Exception as e:
        console.print(f"[red]Error during indexing: {e}[/red]")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ask LLM a question")
    parser.add_argument("--path", required=True, help="Path to project")
    parser.add_argument("--query", required=True,
                        help="Natural language question to ask")
    parser.add_argument(
        "--model", default="deepseek-coder:6.7b", help="LLM model name")
    parser.add_argument("--local", action="store_true",
                        help="Use local Ollama API instead of FastAPI app")
    parser.add_argument("--reindex", action="store_true",
                        help="Force reindexing of the codebase")

    args = parser.parse_args()
    abs_path = os.path.abspath(args.path)

    if not os.path.exists(abs_path):
        console.print(f"[red]Error: Provided path does not exist: {
                      abs_path}[/red]")
        exit(1)

    run_indexer(abs_path, force=args.reindex)

    if args.local:
        if not is_ollama_running():
            console.print(
                "[yellow]Ollama not running. Attempting to start it...[/yellow]")
            if not try_start_ollama():
                return
        ensure_model_available(args.model)

        try:
            context = get_context_from_embeddings(abs_path, args.query)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return

        prompt = build_prompt(context, args.query)
        stream_local_chat(args.model, prompt)
    else:
        stream_remote_chat(abs_path, args.query)


if __name__ == "__main__":
    main()
