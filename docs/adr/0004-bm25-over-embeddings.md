# 4. BM25 over embeddings for the fiscal-rules RAG

- Status: accepted
- Date: 2026-11

## Context

The knowledge base is a few short Markdown notes about access keys, CNPJ, CFOP, CT-e and NFS-e/ISS. Questions are full of exact identifiers (`CFOP 6102`, `modelo 57`, `tomador`, `módulo 11`) in English and Portuguese. The project must run offline in CI with no model download.

## Decision

Retrieval uses BM25 over heading-aware chunks, with accent folding, and answers are extractive with citations to the section they came from. The retriever sits behind `KnowledgeBase.search(question, k)`.

## Consequences

- Exact identifiers match exactly, ranking is explainable, and there is no embedding model, vector store or GPU dependency.
- Measured on 20 hand-written questions: hit@1 85%, hit@3 100%, MRR 0.917.
- Paraphrases with no shared terms are missed; a larger or less technical corpus would justify a hybrid (BM25 + embeddings) index behind the same interface.
