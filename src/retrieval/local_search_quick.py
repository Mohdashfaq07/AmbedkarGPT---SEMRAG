"""
local_search_quick.py
Implements Equation (4) local RAG retrieval using cosine similarity
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LocalRetriever:
    def __init__(self, chunks_path: str, embed_model="all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(embed_model)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.texts = [c["text"] for c in self.chunks]

        print("Embedding chunks...")
        self.chunk_embeddings = self.embed_model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def search(self, query: str, top_k=5):
        query_emb = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = cosine_similarity(query_emb, self.chunk_embeddings)[0]
        idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "chunk_id": int(self.chunks[i]["id"]),
                "text": self.chunks[i]["text"],
                "score": float(scores[i])
            }
            for i in idx
        ]


if __name__ == "__main__":
    retriever = LocalRetriever("../../data/processed/chunks.json")
    print(retriever.search("What did Ambedkar say about caste?"))
