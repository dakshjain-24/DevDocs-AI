import json
import pickle
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi


load_dotenv()


# PATHS

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
LOCAL_DOCS_DIR = DATA_DIR / "local_docs"

URL_FILE = DATA_DIR / "docs_urls.txt"

STORAGE_DIR = BASE_DIR / "storage"
FAISS_DIR = STORAGE_DIR / "faiss_index"

BM25_PATH = STORAGE_DIR / "bm25.pkl"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"


# CONFIG

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# CLEAN TEXT

def clean_text(text: str) -> str:

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# BM25 TOKENIZATION

def tokenize(text: str):

    return re.findall(
        r"[A-Za-z0-9_\.]+",
        text.lower()
    )


# FETCH WEB DOCUMENT

def fetch_web_document(url: str) -> Document:

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DevDocsRAG/1.0)"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
        allow_redirects=True
    )

    response.raise_for_status()

    final_url = response.url

    if final_url.rstrip("/") != url.rstrip("/"):
        print("\nRedirect detected:")
        print(f"Original: {url}")
        print(f"Final: {final_url}")

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "aside"
    ]):
        tag.decompose()

    content = soup.find("main") or soup.find("article") or soup.body

    if content is None:
        raise ValueError(
            f"Could not extract useful content from {final_url}"
        )

    text = content.get_text(separator="\n", strip=True)
    text = clean_text(text)

    if not text:
        raise ValueError(
            f"Extracted page was empty: {final_url}"
        )

    title = final_url

    if soup.title:
        title_text = soup.title.get_text(strip=True)

        if title_text:
            title = title_text

    return Document(
        page_content=text,
        metadata={
            "title": title,
            "source": final_url,
            "original_source": url,
            "source_type": "web"
        }
    )


# LOAD URL LIST

def load_web_documents():

    documents = []
    failures = []

    if not URL_FILE.exists():
        print(f"Warning: URL file not found: {URL_FILE}")
        return documents, failures

    urls = []

    with open(URL_FILE, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if line and not line.startswith("#"):
                urls.append(line)

    print(f"Found {len(urls)} documentation URLs.")

    for index, url in enumerate(urls, start=1):

        print(f"\n[{index}/{len(urls)}] Fetching: {url}")

        try:
            document = fetch_web_document(url)
            documents.append(document)

            print(f"Loaded: {document.metadata.get('title')}")

        except Exception as error:

            print(f"FAILED: {url}")
            print(f"Reason: {error}")

            failures.append({
                "source": url,
                "error": str(error)
            })

    return documents, failures


# LOAD LOCAL DOCUMENTS

def load_local_documents():

    documents = []

    LOCAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".txt", ".md"}

    for path in LOCAL_DOCS_DIR.rglob("*"):

        if path.is_file() and path.suffix.lower() in supported_extensions:

            try:
                text = path.read_text(encoding="utf-8")
                text = clean_text(text)

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "title": path.stem,
                            "source": str(path.resolve()),
                            "original_source": str(path.resolve()),
                            "source_type": "local"
                        }
                    )
                )

                print(f"Loaded local file: {path.name}")

            except Exception as error:
                print(f"Failed local document {path}: {error}")

    return documents


# DEDUPLICATE DOCUMENTS

def deduplicate_documents(documents):

    unique_documents = []
    seen_sources = set()
    duplicates = []

    for document in documents:

        source = document.metadata.get("source", "").rstrip("/")

        if source in seen_sources:

            duplicates.append(
                document.metadata.get("original_source", source)
            )

            continue

        seen_sources.add(source)
        unique_documents.append(document)

    if duplicates:

        print(f"\nRemoved {len(duplicates)} duplicate redirected document(s).")

        for duplicate in duplicates:
            print(f"Duplicate: {duplicate}")

    return unique_documents


# ADD CHUNK METADATA

def add_chunk_metadata(chunks):

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


# MAIN

def main():

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n================================")
    print("LOADING DOCUMENTS")
    print("================================\n")

    web_documents, failures = load_web_documents()
    local_documents = load_local_documents()

    raw_documents = web_documents + local_documents

    if not raw_documents:
        raise RuntimeError(
            "No documents were loaded.\n"
            "Check data/docs_urls.txt or data/local_docs/."
        )

    print(f"\nDocuments before deduplication: {len(raw_documents)}")

    raw_documents = deduplicate_documents(raw_documents)

    print(f"Documents after deduplication: {len(raw_documents)}")

    print("\n================================")
    print("CHUNKING DOCUMENTS")
    print("================================\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(raw_documents)
    chunks = add_chunk_metadata(chunks)

    if not chunks:
        raise RuntimeError("Chunking produced zero chunks.")

    print(f"Created {len(chunks)} chunks.")

    print("\n================================")
    print("LOADING EMBEDDING MODEL")
    print("================================\n")

    print(f"Embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("\n================================")
    print("BUILDING FAISS INDEX")
    print("================================\n")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(str(FAISS_DIR))

    print(f"FAISS index saved to: {FAISS_DIR}")

    print("\n================================")
    print("BUILDING BM25 INDEX")
    print("================================\n")

    tokenized_corpus = [
        tokenize(document.page_content)
        for document in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as file:

        pickle.dump(
            {
                "bm25": bm25,
                "documents": chunks
            },
            file
        )

    print(f"BM25 index saved to: {BM25_PATH}")

    print("\n================================")
    print("SAVING MANIFEST")
    print("================================\n")

    manifest = {
        "documents": len(raw_documents),
        "chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,

        "sources": [
            {
                "title": document.metadata.get("title", ""),
                "source": document.metadata.get("source", ""),
                "original_source": document.metadata.get("original_source", ""),
                "source_type": document.metadata.get("source_type", "")
            }
            for document in raw_documents
        ],

        "failures": failures
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Manifest saved to: {MANIFEST_PATH}")

    print("\n================================")
    print("INGESTION COMPLETE")
    print("================================")

    print(f"Documents : {len(raw_documents)}")
    print(f"Chunks    : {len(chunks)}")
    print(f"Failures  : {len(failures)}")
    print(f"FAISS     : {FAISS_DIR}")
    print(f"BM25      : {BM25_PATH}")

    print("================================\n")


if __name__ == "__main__":
    main()