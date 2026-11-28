"""Small RAG over fiscal-rule notes: chunking, BM25 retrieval, cited answers."""

from docai.rag.evaluate import evaluate_retrieval, load_qa_set
from docai.rag.knowledge import Answer, Chunk, KnowledgeBase, Passage, chunk_markdown

__all__ = [
    "Answer",
    "Chunk",
    "KnowledgeBase",
    "Passage",
    "chunk_markdown",
    "evaluate_retrieval",
    "load_qa_set",
]
