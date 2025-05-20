import argparse
import os
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb

EXTENSIONS = [".py"]  # Customize as needed


def get_hash(path: str) -> str:
    """Hash path string to detect changes"""
    return hashlib.sha256(path.encode()).hexdigest()


def load_chunks(source_dir: str, chunk_size=800):
    chunks = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if any(f.endswith(ext) for ext in EXTENSIONS):
                with open(os.path.join(root, f), encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        chunks.append(chunk)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the source directory")
    args = parser.parse_args()
    path = os.path.abspath(args.path)
    path_hash = get_hash(path)

    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection_name = "project_index_" + path_hash[:10]

    # Always delete and recreate collection
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"🗑️ Deleted existing collection for: {path}")
    except Exception:
        pass  # Collection may not exist yet

    collection = chroma_client.get_or_create_collection(name=collection_name)

    print(f"Indexing source code at {path}...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_chunks(path)
    embeddings = model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    print(f"✅ Indexed {len(chunks)} chunks.")
    print(f"All existing collections: {chroma_client.list_collections()}")


if __name__ == "__main__":
    main()
