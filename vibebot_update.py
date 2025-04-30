import os
import glob
import json
from git import Repo
from tree_sitter import Language, Parser
import tree_sitter_c_sharp as csharp
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.DEBUG)
# --- Setup ---
load_dotenv()
device = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
print(f"Using device: {device}")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
repo_url = os.getenv("REPO_URL", "https://github.com/Civ13/Civ14.git")

# Initialize C# parser
CSHARP_LANGUAGE = Language(csharp.language())


def chunk_file(file_path):
    try:
        parser = Parser(language=CSHARP_LANGUAGE)
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        tree = parser.parse(code.encode())
        chunks = [
            code[node.start_byte : node.end_byte]
            for node in tree.root_node.children
            if node.type
            in ["class_declaration", "method_declaration", "property_declaration"]
        ]
        if not chunks:
            step = 500  # Smaller chunks for metadata
            overlap = 100
            chunks = [code[i : i + step] for i in range(0, len(code), step - overlap)]
        # Truncate chunks for metadata (Pinecone limit: ~40 KB)
        return [(file_path, chunk[:500]) for chunk in chunks]  # Limit to 500 chars
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return []


def main():
    # Clone repo (sparse checkout)
    local_path = "data"
    sparse_checkout_file = "sparse_checkout.txt"
    if not os.path.exists(local_path):
        repo = Repo.init(local_path)
        repo.git.config("core.sparseCheckout", "true")
        os.makedirs(f"{local_path}/.git/info", exist_ok=True)
        with open(sparse_checkout_file, "r") as src, open(
            f"{local_path}/.git/info/sparse-checkout", "w"
        ) as dst:
            dst.write(src.read())
        repo.git.remote("add", "origin", repo_url)
        repo.git.fetch("origin")
        repo.git.checkout("main")

    include_patterns = ["**/*.cs", "**/*.csproj", "**/*.md", "**/*.yml", "**/*.py"]
    files = [
        f
        for pattern in include_patterns
        for f in glob.glob(f"data/{pattern}", recursive=True)
        if "/bin/" not in f and "/obj/" not in f and "/RobustToolbox/" not in f
    ]
    print(f"Total files: {len(files)}")

    with Pool(processes=4) as pool:
        results = pool.map(chunk_file, files)

    code_chunks = [chunk for result in results for _, chunk in result]
    metadata = [
        {"file": file, "chunk": chunk} for result in results for file, chunk in result
    ]
    print(f"Total chunks: {len(code_chunks)}")

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    model = HuggingFaceEmbeddings(
        model_name=embed_model, model_kwargs={"device": device}
    )

    embeddings = model.embed_documents(code_chunks)

    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=len(embeddings[0]),
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(pinecone_index_name)

    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch_vectors = [
            (
                str(j),
                embeddings[j],
                {"chunk": code_chunks[j][:500], "file": metadata[j]["file"]},
            )
            for j in range(i, min(i + batch_size, len(embeddings)))
        ]
        index.upsert(vectors=batch_vectors)

    print(f"Upserted batch {i // batch_size + 1}")


if __name__ == "__main__":
    main()
