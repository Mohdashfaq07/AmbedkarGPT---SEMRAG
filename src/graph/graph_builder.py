"""
graph_builder.py
Builds Knowledge Graph from extracted entities and chunk links.
Compatible with NetworkX 3.x
"""

import json
from pathlib import Path
import networkx as nx
from tqdm import tqdm
from src.graph.entity_extractor import extract_entities  # absolute import

def build_graph(chunks_json_path: str, out_graph_path: str, out_meta_json: str):
    """
    Construct a knowledge graph from chunk entities.
    Nodes = entities + chunks
    Edges = entity appears in chunk; shared entities connect chunks
    """

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    G = nx.Graph()
    last_entity = None

    for item in tqdm(chunks, desc="Building Knowledge Graph..."):
        chunk_id = item["id"]
        chunk_text = item["text"]

        # Add chunk node
        G.add_node(f"chunk_{chunk_id}", type="chunk", text=chunk_text)

        # Extract and add entities
        entities = extract_entities(chunk_text)

        for ent in entities:
            ent_name = ent["text"]
            G.add_node(ent_name, type="entity", label=ent["label"])

            # Entity → Chunk link
            G.add_edge(ent_name, f"chunk_{chunk_id}", relation="mentions")

            # Connect consecutive entities to strengthen relation
            if last_entity:
                G.add_edge(last_entity, ent_name, relation="related")

            last_entity = ent_name

    # Save graph (compatible with NetworkX 3.x)
    Path(out_graph_path).parent.mkdir(parents=True, exist_ok=True)
    nx.readwrite.gpickle.write_gpickle(G, out_graph_path)

    meta = {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "chunks_indexed": len(chunks)
    }
    Path(out_meta_json).write_text(json.dumps(meta, indent=2))
    print(f"Saved graph to {out_graph_path}; meta -> {out_meta_json}")
    return G


if __name__ == "__main__":
    build_graph(
        "../../data/processed/chunks.json",
        "../../data/processed/knowledge_graph.pkl",
        "../../data/processed/knowledge_graph_meta.json"
    )
