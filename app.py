import json
from pathlib import Path

import streamlit as st

from rag import (
    load_vectorstore,
    load_bm25,
    get_reranker,
    get_model,
    retrieve,
    answer_question
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    BASE_DIR
    / "storage"
    / "manifest.json"
)


st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# LOAD RESOURCES
# =========================================================

@st.cache_resource
def load_resources():

    vectorstore = load_vectorstore()

    bm25, bm25_documents = (
        load_bm25()
    )

    reranker = get_reranker()

    model = get_model()

    return (
        vectorstore,
        bm25,
        bm25_documents,
        reranker,
        model
    )


# =========================================================
# MANIFEST
# =========================================================

def load_manifest():

    if not MANIFEST_PATH.exists():
        return None

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# HEADER
# =========================================================

st.title("📚 DevDocs AI")

st.caption(
    "Technical Documentation RAG • "
    "FAISS + BM25 + RRF + Cross-Encoder"
)


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

try:

    (
        vectorstore,
        bm25,
        bm25_documents,
        reranker,
        model
    ) = load_resources()

except Exception as error:

    st.error(
        "Could not load the knowledge base."
    )

    st.code(str(error))

    st.info(
        "Run `python ingest.py` first."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Retrieval Settings"
)


dense_k = st.sidebar.slider(
    "Dense candidates",
    min_value=5,
    max_value=30,
    value=10
)


bm25_k = st.sidebar.slider(
    "BM25 candidates",
    min_value=5,
    max_value=30,
    value=10
)


hybrid_k = st.sidebar.slider(
    "Hybrid candidates",
    min_value=5,
    max_value=20,
    value=10
)


final_k = st.sidebar.slider(
    "Max chunks sent to LLM",
    min_value=2,
    max_value=8,
    value=4
)


# =========================================================
# KNOWLEDGE BASE INFORMATION
# =========================================================

manifest = load_manifest()

if manifest:

    st.sidebar.divider()

    st.sidebar.subheader(
        "Knowledge Base"
    )

    st.sidebar.write(
        f"Documents: "
        f"{manifest.get('documents', 0)}"
    )

    st.sidebar.write(
        f"Chunks: "
        f"{manifest.get('chunks', 0)}"
    )

    with st.sidebar.expander(
        "Indexed sources"
    ):

        for source in manifest.get(
            "sources",
            []
        ):

            st.write(
                source.get(
                    "title",
                    "Untitled"
                )
            )

            st.caption(
                source.get(
                    "source",
                    ""
                )
            )


# =========================================================
# QUESTION
# =========================================================

question = st.text_area(
    "Ask a technical documentation question",
    height=130,
    placeholder=(
        "Example: What is RunnableParallel "
        "and how is it used?"
    )
)


# =========================================================
# RUN RAG
# =========================================================

if st.button(
    "Ask",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Enter a question first."
        )

        st.stop()

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    with st.spinner(
        "Retrieving relevant documentation..."
    ):

        results = retrieve(
            vectorstore=vectorstore,
            bm25=bm25,
            bm25_documents=bm25_documents,
            reranker=reranker,
            query=question,
            dense_k=dense_k,
            bm25_k=bm25_k,
            hybrid_k=hybrid_k,
            final_k=final_k
        )

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    with st.spinner(
        "Generating grounded answer..."
    ):

        try:

            answer = answer_question(
                question=question,
                results=results,
                model=model
            )

        except Exception as error:

            st.error(
                "LLM generation failed."
            )

            st.code(str(error))

            st.stop()

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    st.subheader("Answer")

    st.markdown(answer)


    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "Retrieved Sources"
    )

    for index, item in enumerate(
        results,
        start=1
    ):

        document = item["document"]

        title = document.metadata.get(
            "title",
            "Unknown source"
        )

        source = document.metadata.get(
            "source",
            ""
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "?"
        )

        with st.expander(
            f"Source {index}: {title}"
        ):

            st.write(
                f"**Chunk ID:** {chunk_id}"
            )

            if source.startswith(
                "http"
            ):

                st.markdown(
                    f"**URL:** {source}"
                )

            else:

                st.write(
                    f"**Source:** {source}"
                )

            st.write(
                "**Retrieval information**"
            )

            st.write(
                "Dense rank:",
                item.get(
                    "dense_rank",
                    "Not retrieved"
                )
            )

            st.write(
                "BM25 rank:",
                item.get(
                    "bm25_rank",
                    "Not retrieved"
                )
            )

            if (
                item.get(
                    "bm25_score"
                )
                is not None
            ):

                st.write(
                    "BM25 score:",
                    round(
                        item[
                            "bm25_score"
                        ],
                        4
                    )
                )

            st.write(
                "RRF score:",
                round(
                    item[
                        "rrf_score"
                    ],
                    6
                )
            )

            st.write(
                "Reranker score:",
                round(
                    item[
                        "reranker_score"
                    ],
                    4
                )
            )

            st.divider()

            st.write(
                document.page_content
            )