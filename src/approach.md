# SEM-RAG AmbedkarGPT — Approach

This project implements a Semantic Retrieval-Augmented Generation (SEM-RAG) system
inspired by the SemRAG research paper.

## Key Components
1. Semantic Chunking Algorithm
   - PDF text is split into semantically coherent segments using embedding 
     similarity and merging windows.
   
2. Knowledge Graph Construction
   - Named Entities and relationships extracted from chunks
   - Graph stored in NetworkX (.pkl)
   - Louvain community detection produces high-level topic clusters

3. Retrieval Strategy
   - Local Search (Equation 4) = chunk-level cosine similarity vs query
   - Global Search (Equation 5) = community relevance scoring
   - Hybrid ranking selects final top-K chunks

4. LLM Generation
   - Local LLM through Ollama (TinyLLama / Mistral / LLaMA3)
   - Prompt templates enforce context-only answer generation

5. Explainability
   - Show top chunks used + their similarity score

## Why This is Better Than Standard RAG
- Traditional RAG retrieves with only vector similarity
- SemRAG adds structured reasoning via knowledge graph & communities 
- Better context selection, reduced hallucination, higher confidence

## Evaluation
Queries tested:
- Why did Ambedkar oppose the caste system?
- Why did Ambedkar convert to Buddhism?
- What were Ambedkar’s views on the Constitution?

Performance validated based on contextual accuracy and minimal hallucination.

## Future Enhancements
- Streamlit UI
- Multi-lingual support (Hindi/Marathi)
- Fine-tuned embeddings for legal/Indian philosophical text
