import pickle
import sys

BM25_PATH = "storage/bm25.pkl"

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")


def main():
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    documents = data["documents"]

    print(f"Total chunks: {len(documents)}\n")

    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id")
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")

        print("=" * 100)
        print(f"CHUNK ID: {chunk_id}")
        print(f"TITLE: {title}")
        print(f"SOURCE: {source}")
        print("-" * 100)
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()