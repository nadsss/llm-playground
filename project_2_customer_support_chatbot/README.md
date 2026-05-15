# Project 2: Customer-Support Chatbot for an E-Commerce Store

This project implements a simple Retrieval-Augmented Generation (RAG)-style support bot using:
- `sentence-transformers` for embeddings
- `FAISS` for similarity search
- a local FAQ knowledge base
- optional Streamlit UI

## Folder Structure

- `kb/store_faq.txt` - store policies and support information
- `rag_chatbot.py` - CLI chatbot
- `streamlit_app.py` - web UI
- `requirements.txt` - dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Run CLI Chatbot

```bash
python rag_chatbot.py
```

## Run Streamlit App

```bash
streamlit run streamlit_app.py
```

## Example Questions

- "How long does shipping take?"
- "Can I return an opened item?"
- "How do refunds work?"
- "What payment methods are accepted?"
