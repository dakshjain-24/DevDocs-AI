import os
import pickle
import re
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from sentence_transformers import CrossEncoder


load_dotenv()


# PATHS

BASE_DIR = Path(__file__).resolve().parent

STORAGE_DIR = BASE_DIR / "storage"

FAISS_DIR = STORAGE_DIR / "faiss_index"

BM25_PATH = STORAGE_DIR / "bm25.pkl"


# MODELS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# TOKENIZER

def tokenize(text: str):

    return re.findall(
        r"[A-Za-z0-9_\.]+",
        text.lower()
    )


# LOAD EMBEDDINGS

def get_embeddings():

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# LOAD FAISS

def load_vectorstore():

    if not FAISS_DIR.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python ingest.py` first."
        )

    embeddings = get_embeddings()

    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


# LOAD BM25

def load_bm25():

    if not BM25_PATH.exists():
        raise FileNotFoundError(
            "BM25 index not found. Run `python ingest.py` first."
        )

    with open(BM25_PATH, "rb") as file:
        data = pickle.load(file)

    return data["bm25"], data["documents"]


# DENSE / FAISS RETRIEVAL

def retrieve_dense(vectorstore, query, k=10):

    results = vectorstore.similarity_search_with_score(query, k=k)

    formatted_results = []

    for document, distance in results:

        formatted_results.append({
            "document": document,
            "dense_distance": float(distance)
        })

    return formatted_results


# BM25 RETRIEVAL

def retrieve_bm25(bm25, documents, query, k=10):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "document": documents[index],
            "bm25_score": float(scores[index])
        })

    return results


# RECIPROCAL RANK FUSION

def reciprocal_rank_fusion(dense_results, bm25_results, final_k=10, rrf_k=60):

    """
    Combines FAISS and BM25 rankings.

    RRF score = 1 / (rrf_k + rank)
    """

    scores = {}
    documents = {}
    metadata = {}


    # Dense rankings

    for rank, item in enumerate(dense_results, start=1):

        document = item["document"]
        chunk_id = document.metadata["chunk_id"]

        documents[chunk_id] = document

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

        metadata.setdefault(chunk_id, {})

        metadata[chunk_id]["dense_rank"] = rank
        metadata[chunk_id]["dense_distance"] = item["dense_distance"]


    # BM25 rankings

    for rank, item in enumerate(bm25_results, start=1):

        document = item["document"]
        chunk_id = document.metadata["chunk_id"]

        documents[chunk_id] = document

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

        metadata.setdefault(chunk_id, {})

        metadata[chunk_id]["bm25_rank"] = rank
        metadata[chunk_id]["bm25_score"] = item["bm25_score"]


    # Sort

    ranked_ids = sorted(
        scores.keys(),
        key=lambda chunk_id: scores[chunk_id],
        reverse=True
    )

    final_results = []

    for chunk_id in ranked_ids[:final_k]:

        item = {
            "document": documents[chunk_id],

            "rrf_score": scores[chunk_id],

            "dense_rank": metadata[chunk_id].get("dense_rank"),

            "bm25_rank": metadata[chunk_id].get("bm25_rank"),

            "dense_distance": metadata[chunk_id].get("dense_distance"),

            "bm25_score": metadata[chunk_id].get("bm25_score")
        }

        final_results.append(item)

    return final_results


# CROSS ENCODER

def get_reranker():

    return CrossEncoder(RERANKER_MODEL)


def rerank(query, results, reranker, final_k=4, min_score=0.0):

    if not results:
        return []

    pairs = [
        (
            query,
            item["document"].page_content
        )
        for item in results
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for item, score in zip(results, scores):

        new_item = item.copy()

        new_item["reranker_score"] = float(score)

        reranked_results.append(new_item)

    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    reranked_results = [
        item
        for item in reranked_results
        if item["reranker_score"] >= min_score
    ]

    return reranked_results[:final_k]


# RETRIEVAL RUNNABLE

def build_retrieval_chain(vectorstore, bm25, bm25_documents, reranker, dense_k=10, bm25_k=10, hybrid_k=10, final_k=4):

    dense = RunnableLambda(
        lambda query: retrieve_dense(vectorstore, query, k=dense_k)
    )

    sparse = RunnableLambda(
        lambda query: retrieve_bm25(bm25, bm25_documents, query, k=bm25_k)
    )

    parallel = RunnableParallel(
        dense=dense,
        sparse=sparse,
        query=RunnablePassthrough()
    )

    rrf = RunnableLambda(
        lambda x: {
            "query": x["query"],
            "results": reciprocal_rank_fusion(x["dense"], x["sparse"], final_k=hybrid_k)
        }
    )

    reranker_chain = RunnableLambda(
        lambda x: rerank(
            x["query"], x["results"], reranker, final_k=final_k
        )
    )

    return parallel | rrf | reranker_chain


# COMPLETE RETRIEVAL PIPELINE

def retrieve(vectorstore, bm25, bm25_documents, reranker, query, dense_k=10, bm25_k=10, hybrid_k=10, final_k=4):

    chain = build_retrieval_chain(vectorstore, bm25, bm25_documents, reranker, dense_k, bm25_k, hybrid_k, final_k)

    return chain.invoke(query)


# LLM

def get_model():

    repo_id = os.getenv(
        "HF_LLM_REPO",
        "meta-llama/Llama-3.1-8B-Instruct"
    )

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=650,
        temperature=0.15
    )

    return ChatHuggingFace(llm=llm)


# PROMPT

ANSWER_PROMPT = PromptTemplate(
    template="""
You are DevDocs AI, a technical documentation assistant.

Relevant documentation has already been retrieved and filtered
for the user's question.

Answer the user's question ONLY using the documentation below.

STRICT RULES:

1. Every factual claim must be directly supported by the
   retrieved documentation.

2. Do NOT use outside knowledge, assumptions, guesses, or
   information from your training data.

3. Do NOT invent APIs, classes, methods, parameters, imports,
   syntax, code examples, or version-specific behavior.

4. You may include a code example ONLY when the retrieved
   documentation contains enough information to support that
   example.

5. Answer only what can be supported by the provided context.
   If the context supports only part of the question, answer
   only that part.

6. Cite supporting chunks using [Source 1], [Source 2], etc.

Retrieved documentation:

{context}

User Question:

{question}

Answer:
""",
    input_variables=["context", "question"]
)


# FORMAT CONTEXT

def format_context(results):
    formatted = []

    for index, item in enumerate(results, start=1):
        document = item["document"]
        title = document.metadata.get("title", "Unknown")
        source = document.metadata.get("source", "Unknown")

        formatted.append(
            f"[Source {index}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted)


# ANSWER GENERATION

def answer_question(question, results, model):

    if not results:
        return (
            "The retrieved documentation does not contain enough "
            "information to answer this question confidently."
        )

    context = format_context(results)

    chain = ANSWER_PROMPT | model | StrOutputParser()

    return chain.invoke({"context": context, "question": question})