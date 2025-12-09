"""
run_demo.py

Driver script that:
- Runs semantic chunking if needed
- Builds knowledge graph if needed
- Runs community detection if needed
- Starts an interactive loop: retrieve top chunks, call Ollama, print answer + citations

Usage:
    python run_demo.py --pdf data/Ambedkar_book.pdf --model mistral
"""

import argparse
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
import traceback

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def ensure_chunks(pdf_path, chunks_path):
    if chunks_path.exists():
        print(f"[+] Chunks already exist at {chunks_path}")
        return
    print("[*] Running semantic chunking (this may take a few minutes)...")
    chunk_mod = load_module_from_path("semantic_chunking", SRC / "chunking" / "semantic_chunking.py")
    # Build args namespace similar to argparse
    args = SimpleNamespace(
        pdf_path=str(pdf_path),
        out_path=str(chunks_path),
        embed_model="all-MiniLM-L6-v2",
        batch_size=64,
        buffer_size=1,
        theta=0.35,
        max_tokens=1024,
        overlap_tokens=128
    )
    try:
        chunk_mod.main(args)
    except Exception as e:
        print("Error during chunking:")
        traceback.print_exc()
        sys.exit(1)

def ensure_graph(chunks_path, graph_path, meta_path):
    if graph_path.exists():
        print(f"[+] Knowledge graph already exists at {graph_path}")
        return
    print("[*] Building knowledge graph...")
    graph_mod = load_module_from_path("graph_builder", SRC / "graph" / "graph_builder.py")
    try:
        # graph_builder.build_graph(chunks_json_path, out_pkl, out_meta_json)
        graph_mod.build_graph(str(chunks_path), str(graph_path), str(meta_path))
    except Exception as e:
        print("Error during graph build:")
        traceback.print_exc()
        sys.exit(1)

def ensure_communities(graph_path, communities_path, model_name):
    if communities_path.exists():
        print(f"[+] Communities already exist at {communities_path}")
        return
    print("[*] Running community detection & summarization (calls Ollama; this may take time)...")
    comm_mod = load_module_from_path("community_detector", SRC / "graph" / "community_detector.py")
    try:
        comm_mod.detect_communities(str(graph_path), str(communities_path), model_name=str(model_name))
    except Exception as e:
        print("Error during community detection:")
        traceback.print_exc()
        sys.exit(1)

def interactive_loop(graph_path, chunks_path, model_name):
    # Load retriever
    retriever_mod = load_module_from_path("local_search_quick", SRC / "retrieval" / "local_search_quick.py")
    try:
        retriever = retriever_mod.LocalRetriever(str(chunks_path))
    except Exception as e:
        print("Failed to initialize retriever (embedding chunks).")
        traceback.print_exc()
        sys.exit(1)

    # Load Ollama client
    ollama_mod = load_module_from_path("ollama_client", SRC / "llm" / "ollama_client.py")
    try:
        client = ollama_mod.OllamaClient(model=model_name)
    except Exception as e:
        print("Failed to initialize Ollama client.")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== AmbedkarGPT — interactive demo ===")
    print("Type a question and press Enter. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            q = input("Question: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            print("\n[1/2] Retrieving top chunks...")
            results = retriever.search(q, top_k=2)

            # Build context with chunk ids + short text
            context_parts = []
            for r in results:
                cid = r["chunk_id"]
                text = r["text"]
                score = r["score"]
                # Keep each chunk reasonably short in the prompt (truncate if huge)
                preview = text[:500]  # reduce to 500 characters
                context_parts.append(f"{preview}")


            context = "\\n\\n---\\n\\n".join(context_parts)
            prompt = (
                "Answer the following question using ONLY the context below.\n"
                "Write 3-4 sentences in clear, factual language.\n"
                "If the context does not contain the answer, respond with: 'Information not available in context.'\n\n"
                f"Context:\n{context[:800]}\n\n"
                f"Question: {q}\n\n"
                "Answer:"
            )

            print("[2/2] Generating answer from Ollama (this may take a few seconds)...")
            answer = client.generate(prompt)

            print("\\n=== Answer ===")
            print(answer)
            print("\\n=== Top chunks used ===")
            for r in results:
                print(f"[chunk:{r['chunk_id']}] score={r['score']:.3f}")
            print("\n---\n")

        except KeyboardInterrupt:
            print("\\nInterrupted. Exiting.")
            break
        except Exception as e:
            print("Error during query handling:")
            traceback.print_exc()
            print("Continuing...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default="data/Ambedkar_book.pdf")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model name")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}. Place 'Ambedkar_book.pdf' inside the 'data' folder.")
        sys.exit(1)

    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    graph_path = ROOT / "data" / "processed" / "knowledge_graph.pkl"
    meta_path = ROOT / "data" / "processed" / "knowledge_graph_meta.json"
    communities_path = ROOT / "data" / "processed" / "communities.json"

    ensure_chunks(pdf_path, chunks_path)
    ensure_graph(chunks_path, graph_path, meta_path)
    ensure_communities(graph_path, communities_path, args.model)

    interactive_loop(graph_path, chunks_path, args.model)


if __name__ == "__main__":
    main()
