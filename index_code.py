# index_code.py
import argparse
import os
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
import pathspec

EXTENSIONS = [".js", ".json", ".md"]
MAX_BATCH_SIZE = 500  # Safe batch size for ChromaDB


def get_hash(path: str) -> str:
    """Hash path string to detect changes"""
    return hashlib.sha256(path.encode()).hexdigest()


def load_gitignore(path: str):
    gitignore_path = os.path.join(path, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            patterns = f.read().splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return None


def is_ignored(filepath: str, spec: pathspec.PathSpec, root_path: str) -> bool:
    rel_path = os.path.relpath(filepath, root_path)
    return spec.match_file(rel_path)


def load_chunks(source_dir: str, chunk_size=800):
    chunks = []
    spec = load_gitignore(source_dir)

    print(f"Extensions to be processed {EXTENSIONS}")
    # files_count = len(os.listdir(source_dir))
    # print(f"Total files count - {files_count}")
    for root, _, files in os.walk(source_dir):
        for f in files:
            full_path = os.path.join(root, f)

            if spec and is_ignored(full_path, spec, source_dir):
                continue

            if any(f.endswith(ext) for ext in EXTENSIONS):
                # print(f"Reading file - {full_path}")
                with open(full_path, encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        chunks.append(chunk)

    return chunks


def add_to_collection_in_batches(collection, documents, embeddings):
    total = len(documents)
    for i in range(0, total, MAX_BATCH_SIZE):
        batch_docs = documents[i:i + MAX_BATCH_SIZE]
        batch_embs = embeddings[i:i + MAX_BATCH_SIZE]
        batch_ids = [f"id_{i+j}" for j in range(len(batch_docs))]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embs,
            ids=batch_ids
        )


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
        pass

    collection = chroma_client.get_or_create_collection(name=collection_name)

    print(f"Indexing source code at {path}...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_chunks(path)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    add_to_collection_in_batches(collection, chunks, embeddings)

    print(f"✅ Indexed {len(chunks)} chunks.")
    print(f"All existing collections: {chroma_client.list_collections()}")


if __name__ == "__main__":
    main()
