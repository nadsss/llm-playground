from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


KB_PATH = Path(__file__).parent / "kb" / "store_faq.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path):
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query, embedder, index, chunks, k=3):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i >= 0:
            results.append((float(score), chunks[i]))
    return results


def synthesize_answer(query, retrieved):
    if not retrieved:
        return "I could not find relevant policy information. Please contact support@example-store.com."

    top_context = "\n\n".join([f"- {chunk.replace(chr(10), ' ')}" for _, chunk in retrieved])
    return (
        f"Question: {query}\n\n"
        "Based on the store policy, here is the best answer:\n"
        f"{top_context}\n\n"
        "If you want, I can provide a shorter final customer reply format."
    )


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(MODEL_NAME)

    print("Loading knowledge base...")
    chunks = load_chunks(KB_PATH)
    index, _ = build_index(chunks, embedder)

    print("Customer Support Chatbot (type 'exit' to quit)")
    while True:
        query = input("\nCustomer question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        retrieved = retrieve(query, embedder, index, chunks, k=3)
        answer = synthesize_answer(query, retrieved)
        print("\n" + answer)


if __name__ == "__main__":
    main()
