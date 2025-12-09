"""
semantic_chunking.py
Implements Algorithm 1 (semantic chunking via embeddings + cosine similarity)
Usage:
  python semantic_chunking.py --pdf_path ../../data/Ambedkar_book.pdf --out_path ../../data/processed/chunks.json
"""
import argparse, json
from pathlib import Path
import spacy
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def split_into_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def buffer_merge(sentences, buffer_size):
    if buffer_size <= 0:
        return sentences[:]
    merged = []
    n = len(sentences)
    for i in range(n):
        start = max(0, i - buffer_size)
        end = min(n, i + buffer_size + 1)
        merged.append(" ".join(sentences[start:end]).strip())
    comp, prev = [], None
    for w in merged:
        if w != prev:
            comp.append(w)
        prev = w
    return comp


def embed_texts(model, texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def split_with_overlap(chunk_text, max_tokens=2000, overlap_tokens=300):
    words = chunk_text.split()
    if len(words) <= max_tokens:
        return [" ".join(words)]
    subs = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        subs.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_tokens)
    return subs


def build_chunks(merged, embeddings, theta, max_tokens, overlap_tokens):
    dots = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    cosine_dists = 1.0 - dots
    chunks = []
    current = [merged[0]]
    for i in range(len(merged) - 1):
        d = float(cosine_dists[i])
        if d < theta:
            current.append(merged[i + 1])
        else:
            sub = split_with_overlap(" ".join(current), max_tokens, overlap_tokens)
            chunks.extend(sub)
            current = [merged[i + 1]]
    if current:
        sub = split_with_overlap(" ".join(current), max_tokens, overlap_tokens)
        chunks.extend(sub)
    return chunks


def save_chunks(chunks, out_path: str):
    out = [{"id": i, "text": c, "word_count": len(c.split())} for i, c in enumerate(chunks)]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out)} chunks to {out_path}")


def main(args):
    print("Reading PDF...")
    text = read_pdf_text(args.pdf_path)
    print("Splitting...")
    sentences = split_into_sentences(text)
    print("Merging...")
    merged = buffer_merge(sentences, args.buffer_size)
    print("Embedding...")
    model = SentenceTransformer(args.embed_model)
    embeddings = embed_texts(model, merged)
    print("Building chunks...")
    chunks = build_chunks(merged, embeddings, args.theta, args.max_tokens, args.overlap_tokens)
    save_chunks(chunks, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--buffer_size", default=1)  # context smoothing
    parser.add_argument("--theta", type=float, default=0.35)
    parser.add_argument("--max_tokens", type=int, default=2000)  # updated
    parser.add_argument("--overlap_tokens", type=int, default=300)  # updated
    args = parser.parse_args()
    main(args)
