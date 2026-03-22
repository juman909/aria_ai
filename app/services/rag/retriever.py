import logging
import time
from typing import Optional

from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.models.session import RAGResult, RetrievedChunk
from app.services.rag.embeddings import EmbeddingService
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGRetriever:
    def __init__(self) -> None:
        self._embedding_service = EmbeddingService()
        self._pinecone = Pinecone(api_key=settings.pinecone_api_key)
        self._index = self._pinecone.Index(settings.pinecone_index_name)
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[dict] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> RAGResult:
        t0 = time.monotonic()

        # Dense retrieval via Pinecone
        query_embedding = await self._embedding_service.embed(query)
        dense_results = self._index.query(
            vector=query_embedding,
            top_k=settings.rag_top_k_retrieval,
            include_metadata=True,
            filter=filters,
        )

        retrieval_ms = int((time.monotonic() - t0) * 1000)

        # Parse Pinecone results
        candidates: list[RetrievedChunk] = []
        for match in dense_results.matches:
            if match.score < settings.rag_score_threshold:
                continue
            candidates.append(
                RetrievedChunk(
                    chunk_id=match.id,
                    text=match.metadata.get("text", ""),
                    score=match.score,
                    metadata={k: v for k, v in match.metadata.items() if k != "text"},
                    source_url=match.metadata.get("source_url", ""),
                )
            )

        if not candidates:
            return RAGResult(chunks=[], retrieval_latency_ms=retrieval_ms)

        # Re-rank with cross-encoder
        t1 = time.monotonic()
        pairs = [[query, c.text] for c in candidates]
        scores = self._reranker.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk.score = float(score)

        candidates.sort(key=lambda x: x.score, reverse=True)
        top_chunks = candidates[:top_k]
        rerank_ms = int((time.monotonic() - t1) * 1000)

        logger.info(
            "rag_retrieval query_len=%d candidates=%d top_k=%d retrieval_ms=%d rerank_ms=%d",
            len(query),
            len(candidates),
            len(top_chunks),
            retrieval_ms,
            rerank_ms,
        )

        return RAGResult(
            chunks=top_chunks,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
        )

    def format_context(self, result: RAGResult) -> str:
        if not result.chunks:
            return ""
        parts = []
        for i, chunk in enumerate(result.chunks, 1):
            doc_name = chunk.metadata.get("doc_name", "Policy Document")
            section = chunk.metadata.get("section", "")
            header = f"[Source {i}: {doc_name}" + (f" — {section}" if section else "") + "]"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n".join(parts)
