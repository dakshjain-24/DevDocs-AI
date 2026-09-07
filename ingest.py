import json
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


# PATHS

BASE_DIR = Path(__file__).resolve().parent

URLS_PATH = BASE_DIR / "data" / "docs_urls.txt"
LOCAL_DOCS_DIR = BASE_DIR / "data" / "local_docs"

STORAGE_DIR = BASE_DIR / "storage"
FAISS_DIR = STORAGE_DIR / "faiss_index"
BM25_PATH = STORAGE_DIR / "bm25.pkl"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"


# CONFIG

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# TOKENIZATION

def tokenize(text):
    return re.findall(r"[A-Za-z0-9_\.]+", text.lower())


# LOAD URLS

def load_urls():

    if not URLS_PATH.exists():
        return []

    with open(URLS_PATH, "r", encoding="utf-8") as file:

        return [
            line.strip()
            for line in file
            if line.strip() and not line.startswith("#")
        ]


# LOAD WEB DOCUMENTS

def load_web_documents(urls):

    documents = []

    for url in urls:

        try:
            print(f"Loading: {url}")

            loader = WebBaseLoader(url)
            docs = loader.load()

            for document in docs:

                document.metadata["source_type"] = "web"

                if not document.metadata.get("title"):
                    document.metadata["title"] = url

                documents.append(document)

        except Exception as error:
            print(f"Failed to load {url}")
            print(error)

    return documents


# LOAD LOCAL DOCUMENTS

def load_local_documents():

    documents = []

    if not LOCAL_DOCS_DIR.exists():
        return documents

    files = list(LOCAL_DOCS_DIR.rglob("*.txt"))
    files += list(LOCAL_DOCS_DIR.rglob("*.md"))

    for path in files:

        try:
            print(f"Loading: {path.name}")

            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()

            for document in docs:

                document.metadata["source"] = str(path)
                document.metadata["source_type"] = "local"
                document.metadata["title"] = path.name

                documents.append(document)

        except Exception as error:
            print(f"Failed to load {path}")
            print(error)

    return documents


# CLEAN DOCUMENTS

def clean_documents(documents):

    cleaned = []

    for document in documents:

        text = document.page_content
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        if not text:
            continue

        document.page_content = text
        cleaned.append(document)

    return cleaned


# DEDUPLICATE

def deduplicate_documents(documents):

    seen = set()
    unique_documents = []

    for document in documents:

        source = document.metadata.get("source", "")

        if source in seen:
            continue

        seen.add(source)
        unique_documents.append(document)

    return unique_documents


# SPLIT DOCUMENTS

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


# BUILD FAISS

def build_faiss(chunks):

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index...")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_DIR))

    return vectorstore


# BUILD BM25

def build_bm25(chunks):

    print("Building BM25 index...")

    tokenized_documents = [
        tokenize(document.page_content)
        for document in chunks
    ]

    bm25 = BM25Okapi(tokenized_documents)

    with open(BM25_PATH, "wb") as file:

        pickle.dump({
            "bm25": bm25,
            "documents": chunks
        }, file)

    return bm25


# SAVE MANIFEST

def save_manifest(documents, chunks):

    sources = []

    for document in documents:

        sources.append({
            "title": document.metadata.get("title", "Untitled"),
            "source": document.metadata.get("source", ""),
            "source_type": document.metadata.get("source_type", "unknown")
        })

    manifest = {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "sources": sources
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


# INGESTION CHAIN

def build_ingestion_chain():

    clean = RunnableLambda(lambda x: {"documents": clean_documents(x["documents"])})

    deduplicate = RunnableLambda(lambda x: {"documents": deduplicate_documents(x["documents"])})

    split = RunnableLambda(
        lambda x: {
            "documents": x["documents"],
            "chunks": split_documents(x["documents"])
        }
    )

    build_indexes = RunnableParallel(
        faiss=RunnableLambda(lambda x: build_faiss(x["chunks"])),
        bm25=RunnableLambda(lambda x: build_bm25(x["chunks"])),
        data=RunnableLambda(lambda x: x)
    )

    return clean | deduplicate | split | build_indexes


# MAIN

def main():

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading documents...\n")

    urls = load_urls()

    web_documents = load_web_documents(urls)
    local_documents = load_local_documents()

    documents = web_documents + local_documents

    print(f"\nLoaded documents: {len(documents)}")

    if not documents:
        print("No documents found.")
        return

    print("\nRunning ingestion pipeline...")

    chain = build_ingestion_chain()

    result = chain.invoke({"documents": documents}) # <-----

    final_documents = result["data"]["documents"]
    chunks = result["data"]["chunks"]

    save_manifest(final_documents, chunks)

    print("\nIngestion complete.")
    print(f"Documents: {len(final_documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"FAISS: {FAISS_DIR}")
    print(f"BM25: {BM25_PATH}")


if __name__ == "__main__":
    main()