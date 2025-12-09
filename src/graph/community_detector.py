"""
community_detector.py
Detects topic-based entity communities using Louvain.
Compatible with NetworkX 3.x
"""

import networkx as nx
import community  # python-louvain
import json
from pathlib import Path
from src.llm.ollama_client import OllamaClient  # absolute import
from tqdm import tqdm

def detect_communities(graph_path: str, out_path: str, model_name: str = 'mistral'):
    G = nx.read_gpickle(graph_path)

    print("Running Louvain community detection...")
    partition = community.best_partition(G)

    # Group nodes by community ID
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    # Build community summaries using Ollama
    community_summaries = {}
    for cid, nodes in tqdm(communities.items(), desc="Summarize communities"):
        texts = []
        for n in nodes:
            mentions = G.nodes[n].get("text", "")
            if mentions:
                texts.append(mentions if len(mentions) < 1000 else mentions[:1000])
        prompt = "Summarize the following passages from Ambedkar's work (2-3 lines):\n\n"
        prompt += "\n\n---\n\n".join(texts[:20])
        try:
            client = OllamaClient(model_name)
            summary = client.generate(prompt)
        except Exception as e:
            summary = f"[ollama-error] {e}"
        community_summaries[str(cid)] = {
            "nodes": nodes,
            "summary": summary
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(community_summaries, ensure_ascii=False, indent=2))
    print(f"Saved {len(community_summaries)} communities to {out_path}")
    return community_summaries


if __name__ == "__main__":
    detect_communities(
        "../../data/processed/knowledge_graph.pkl",
        "../../data/processed/communities.json",
        model_name="mistral"
    )
