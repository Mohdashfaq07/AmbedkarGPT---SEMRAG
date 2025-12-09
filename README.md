# AmbedkarGPT---SEMRAG

**AmbedkarGPT** is a Semantic-Enhanced Retrieval-Augmented Generation (SEM-RAG) system for answering questions from B.R. Ambedkar’s writings. It combines PDF semantic chunking, embeddings, knowledge graph construction, community detection, and a local LLM via Ollama to produce context-aware answers while preventing hallucination.

---

## Features

- **Semantic Chunking**: Converts PDF text into meaningful chunks for retrieval.  
- **Embeddings & Retrieval**: Uses vector similarity search to find relevant chunks.  
- **Knowledge Graph & Communities**: Louvain clustering identifies key themes in the text.  
- **RAG Query Engine**: Retrieves top relevant chunks and generates answers using a local LLM.  
- **Interactive CLI Demo**: Ask questions and get context-based answers.  
- **Offline Execution**: Runs completely locally using Ollama models (`tinyllama`, `gemma`, etc.).  

---

## Getting Started

AmbedkarGPT — SEM-RAG Pipeline Demo

1. Load preprocessed chunks, knowledge graph and community clusters
2. Ask: "Why did Ambedkar oppose the caste system?"
3. System retrieves top semantic chunks
4. Local LLM (TinyLlama) generates context-grounded answer
5. Display context chunks used (traceability and explainability)
6. Ask comparison question e.g. "Why did Ambedkar convert to Buddhism?"
7. Close demo

Commands:
python run_demo.py --model tinyllama --pdf data/Ambedkar_book.pdf

```bash
git clone https://github.com/Mohdashfaq07/AmbedkarGPT---SEMRAG.git
cd AmbedkarGPT---SEMRAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull tinyllama   # or mistral if RAM available
