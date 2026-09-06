# DevDocs AI

### Retrieval-Augmented Technical Documentation Search Engine

DevDocs AI is a **Retrieval-Augmented Generation (RAG)** system designed for question answering over technical documentation. It combines **dense semantic retrieval, sparse lexical retrieval, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and relevance filtering** to retrieve high-quality context before generating grounded answers with source citations.

---

## Overview

Traditional keyword search can struggle with semantically related queries, while dense vector search may miss documents containing important exact terminology.

DevDocs AI combines both approaches:

- **FAISS** for dense semantic retrieval
- **BM25** for sparse keyword-based retrieval
- **Reciprocal Rank Fusion (RRF)** for combining retrieval rankings
- **Cross-Encoder reranking** for improving top-result relevance
- **Relevance filtering** for rejecting unsupported queries
- **LLM-based generation** for producing answers grounded in retrieved documentation

The system also exposes retrieved chunks, document metadata, retrieval ranks, BM25 scores, RRF scores, and reranker scores through a Streamlit interface.

---

## Features

- Web-based technical documentation ingestion
- Local `.txt` and `.md` document ingestion
- Automatic text cleaning and preprocessing
- Recursive document chunking
- Sentence Transformer embeddings
- Persistent FAISS vector index
- BM25 lexical search
- Hybrid dense + sparse retrieval
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- Relevance-based query rejection
- Grounded LLM answer generation
- Source citations for generated answers
- Retrieval diagnostics in the UI
- Offline retrieval evaluation using Hit@K, Recall@K, and MRR

---

## Architecture

```text
                 Technical Documentation
                          |
                          v
                  Document Ingestion
                          |
                          v
                Cleaning & Preprocessing
                          |
                          v
           RecursiveCharacterTextSplitter
                          |
                          v
                       Chunks
                    /          \
                   /            \
                  v              v
       SentenceTransformer      BM25
            Embeddings          Index
                  |              |
                  v              |
             FAISS Index         |
                  |              |
                  +------+-------+
                         |
                         v
              Hybrid Candidate Retrieval
                         |
                         v
             Reciprocal Rank Fusion
                         |
                         v
                Cross-Encoder Reranker
                         |
                         v
                Relevance Filtering
                         |
                  relevant?
                   /     \
                 No       Yes
                 |         |
                 v         v
              Reject     Context
                           |
                           v
                          LLM
                           |
                           v
                Grounded Answer + Sources
```

---

## Retrieval Pipeline

### 1. Dense Retrieval

Documents are converted into vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are stored in a persistent **FAISS** index.

At query time, the question is embedded using the same model and FAISS retrieves semantically similar chunks.

### 2. Sparse Retrieval

A **BM25** index is constructed over the same document chunks.

BM25 complements dense retrieval by performing lexical matching, which is useful when queries contain exact technical terms, API names, classes, or identifiers.

### 3. Reciprocal Rank Fusion

FAISS similarity distances and BM25 scores are not directly comparable.

Instead of combining their raw scores, DevDocs AI combines their rankings using **Reciprocal Rank Fusion (RRF)**:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

This allows results from both retrieval systems to contribute to a unified candidate ranking.

### 4. Cross-Encoder Reranking

The fused candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike the initial retrieval stage, the Cross-Encoder evaluates the query and candidate document together, allowing more precise relevance scoring.

Only the highest-ranked chunks are considered for generation.

---

## Relevance Filtering & Hallucination Control

Retrieving the closest available document does not necessarily mean that the document is actually relevant.

DevDocs AI therefore applies a **Cross-Encoder relevance threshold** after reranking.

If no retrieved chunk passes the configured relevance threshold, the system does not call the LLM with unrelated context and instead returns:

```text
The retrieved documentation does not contain enough information
to answer this question confidently.
```

The current threshold was selected empirically using a small validation set containing both in-domain and out-of-domain queries.

The generation prompt additionally instructs the model to:

- Answer only using retrieved documentation
- Avoid unsupported assumptions
- Avoid inventing APIs, methods, parameters, or code
- Generate code only when supported by retrieved context
- Cite the supporting retrieved chunks

This provides two layers of grounding:

```text
Retrieval-level relevance filtering
              +
Generation-level grounding instructions
```

---

## Evaluation

The retrieval pipeline was evaluated using a manually labelled dataset of **40 in-domain technical-documentation queries**.

The following configurations were compared:

1. FAISS dense retrieval
2. BM25 sparse retrieval
3. FAISS + BM25 + RRF
4. Hybrid retrieval + Cross-Encoder reranking

### Retrieval Results

| Method | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| FAISS only | 0.575 | 0.775 | 0.900 | 0.433 | 0.679 | 0.863 | 0.698 |
| BM25 only | 0.600 | 0.800 | 0.850 | 0.471 | 0.738 | 0.812 | 0.724 |
| FAISS + BM25 + RRF | 0.600 | **0.900** | **0.925** | 0.471 | **0.863** | **0.912** | 0.758 |
| Hybrid + CrossEncoder | **0.800** | 0.875 | 0.900 | **0.646** | 0.825 | 0.875 | **0.850** |

### Observations

Hybrid retrieval using FAISS, BM25, and RRF improved candidate coverage compared with either retrieval method individually.

Cross-Encoder reranking significantly improved top-ranked relevance:

```text
FAISS Hit@1:                 0.575
Hybrid + CrossEncoder Hit@1: 0.800

FAISS MRR:                   0.698
Hybrid + CrossEncoder MRR:   0.850
```

This corresponds to an increase of approximately **39% in Hit@1** and **22% in MRR** relative to the FAISS-only baseline.

RRF achieved the strongest broader candidate coverage, reaching **0.900 Hit@3** and **0.863 Recall@3**, while Cross-Encoder reranking prioritized the most relevant candidates toward the top of the ranking.

---

## Tech Stack

**Language**
- Python

**RAG / LLM**
- LangChain
- Hugging Face

**Retrieval**
- FAISS
- BM25
- Reciprocal Rank Fusion

**Machine Learning**
- Sentence Transformers
- Cross-Encoder
- PyTorch

**Document Processing**
- BeautifulSoup
- RecursiveCharacterTextSplitter

**Frontend**
- Streamlit

---

## Project Structure

```text
devdocs_rag/
|
├── data/
│   ├── local_docs/
│   └── docs_urls.txt
|
├── storage/
│   ├── faiss_index/
│   ├── bm25.pkl
│   └── manifest.json
|
├── app.py
├── ingest.py
├── rag.py
├── evaluate.py
├── evaluate_threshold.py
├── evaluation_dataset.json
├── threshold_dataset.json
├── inspect_chunks.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Main Components

**`ingest.py`**

Loads technical documentation, cleans the extracted text, splits documents into chunks, generates embeddings, and builds the FAISS and BM25 indexes.

**`rag.py`**

Contains the retrieval and generation pipeline including dense retrieval, BM25 retrieval, RRF, Cross-Encoder reranking, relevance filtering, and LLM-based answer generation.

**`app.py`**

Provides the Streamlit interface and displays generated answers, citations, retrieved chunks, and retrieval diagnostics.

**`evaluate.py`**

Benchmarks FAISS, BM25, hybrid RRF retrieval, and Cross-Encoder reranking using Hit@K, Recall@K, and MRR.

**`evaluate_threshold.py`**

Evaluates Cross-Encoder score thresholds using in-domain and out-of-domain questions.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd devdocs_rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

```env
HF_TOKEN=your_huggingface_token
```

Never commit the `.env` file or API tokens to version control.

---

## Ingest Documentation

Add documentation URLs to:

```text
data/docs_urls.txt
```

Local `.txt` and `.md` files can be placed inside:

```text
data/local_docs/
```

Build the indexes using:

```bash
python ingest.py
```

The generated FAISS and BM25 indexes are persisted inside the `storage/` directory.

Re-run ingestion when the documentation, chunking configuration, or embedding model changes.

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application allows you to ask questions about the indexed documentation and inspect the retrieved evidence used to generate each answer.

---

## Evaluation Scripts

Run retrieval evaluation:

```bash
python evaluate.py
```

This compares:

```text
FAISS
BM25
FAISS + BM25 + RRF
Hybrid + CrossEncoder
```

Run relevance-threshold evaluation:

```bash
python evaluate_threshold.py
```

The threshold evaluation compares in-domain acceptance and out-of-domain rejection at different Cross-Encoder score thresholds.

---

## Example Queries

```text
What is a vector store?

What is RecursiveCharacterTextSplitter?

What search types does VectorStoreRetriever support?

How can Hugging Face embedding models be run locally?

What is Text Embeddings Inference?
```

An unrelated query such as:

```text
How does Kubernetes schedule pods?
```

should be rejected when the retrieved documentation does not provide sufficiently relevant evidence.

---

## Limitations

- The current knowledge base contains a limited collection of technical documentation.
- Retrieval quality depends on document coverage and chunking quality.
- Cross-Encoder reranking increases query latency compared with retrieval alone.
- The relevance threshold was selected using a relatively small validation dataset and may require recalibration for a substantially different knowledge base.
- The current evaluation dataset is relatively small and focused on the indexed documentation.
- Web ingestion depends on the structure and accessibility of documentation pages.

---

## Future Improvements

- Expand the technical documentation corpus
- Evaluate retrieval on a larger manually labelled dataset
- Experiment with alternative embedding and reranking models
- Explore query expansion or query rewriting
- Add retrieval and generation latency benchmarks
- Improve document deduplication and incremental indexing
- Add automated tests for ingestion and retrieval components

---

## Author

**Daksh Jain**