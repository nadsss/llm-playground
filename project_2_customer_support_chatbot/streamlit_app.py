import streamlit as st
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


KB_PATH = Path(__file__).parent / "kb" / "store_faq.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_system():
    embedder = SentenceTransformer(MODEL_NAME)
    text = KB_PATH.read_text(encoding="utf-8")
    chunks = [b.strip() for b in text.split("\n\n") if b.strip()]

    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, chunks, index


def retrieve(query, embedder, index, chunks, k=3):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    out = []
    for score, i in zip(scores[0], idxs[0]):
        if i >= 0:
            out.append((float(score), chunks[i]))
    return out


st.title("Week 2: Customer Support Chatbot (RAG)")
st.caption("E-commerce support assistant using local embeddings + FAISS retrieval")

embedder, chunks, index = load_system()
query = st.text_input("Ask a customer-support question")

if query:
    hits = retrieve(query, embedder, index, chunks, k=3)
    st.subheader("Suggested Answer")
    if not hits:
        st.write("I couldn't find a relevant answer. Please contact support@example-store.com")
    else:
        for score, chunk in hits:
            st.write(f"- {chunk.replace(chr(10), ' ')}")
            st.caption(f"relevance: {score:.3f}")
