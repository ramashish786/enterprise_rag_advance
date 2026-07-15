"""Tests for the RAG stack: chunking, embedder, hybrid retrieval, reranker."""
from __future__ import annotations

from uuid import uuid4

import pytest

from app.rag.chunker import chunk_document
from app.rag.embedder import FakeEmbedder, cosine
from app.rag.parser import ParsedDocument, TextBlock
from app.rag.reranker import LexicalReranker
from app.rag.retriever import ChunkRecord, HybridRetriever, RetrievedChunk, bm25_scores


# --- Chunker ---

class TestChunker:
    def test_sections_split_by_heading(self):
        doc = ParsedDocument(
            filename="test.txt",
            blocks=[
                TextBlock(text="Life Insurance", kind="heading", heading_level=1),
                TextBlock(text="Covers death benefit and cash value.", kind="paragraph"),
                TextBlock(text="Vehicle Insurance", kind="heading", heading_level=1),
                TextBlock(text="Covers collision and comprehensive.", kind="paragraph"),
            ],
        )
        parents, children = chunk_document(doc)
        assert len(parents) == 2
        assert "death benefit" in parents[0].content
        assert "collision" in parents[1].content
        assert all(c.parent_id is not None for c in children)

    def test_heading_path_populated(self):
        doc = ParsedDocument(
            filename="t.txt",
            blocks=[
                TextBlock(text="Policies", kind="heading", heading_level=1),
                TextBlock(text="Home", kind="heading", heading_level=2),
                TextBlock(text="A home policy covers the dwelling and contents.", kind="paragraph"),
            ],
        )
        parents, _ = chunk_document(doc)
        assert parents[0].heading_path == ["Policies", "Home"]

    def test_long_parent_split_into_windows(self):
        long_text = " ".join(["Sentence about life insurance coverage."] * 200)
        doc = ParsedDocument(
            filename="t.txt",
            blocks=[TextBlock(text=long_text, kind="paragraph")],
        )
        parents, children = chunk_document(
            doc, child_target_tokens=100, parent_max_tokens=400,
        )
        assert len(parents) > 1
        assert len(children) >= len(parents)


# --- Embedder ---

class TestEmbedder:
    @pytest.mark.asyncio
    async def test_fake_embedder_is_deterministic(self):
        e = FakeEmbedder(dim=128)
        v1 = await e.embed_one("hello world")
        v2 = await e.embed_one("hello world")
        assert v1 == v2
        assert len(v1) == 128

    @pytest.mark.asyncio
    async def test_similar_texts_have_higher_similarity(self):
        e = FakeEmbedder(dim=256)
        v_life = await e.embed_one("life insurance policy coverage")
        v_life2 = await e.embed_one("life insurance policy")
        v_car = await e.embed_one("car engine transmission")
        assert cosine(v_life, v_life2) > cosine(v_life, v_car)


# --- BM25 ---

class TestBM25:
    def test_exact_match_scores_highest(self):
        corpus = [
            "Life insurance covers death benefits.",
            "Vehicle collision coverage.",
            "Home flood damage protection.",
        ]
        scores = bm25_scores("life insurance", corpus)
        # Doc 0 should score highest
        best = max(range(len(scores)), key=lambda i: scores[i])
        assert best == 0


# --- Hybrid retriever + RBAC pre-filter ---

@pytest.mark.asyncio
class TestHybridRetriever:
    async def _make_corpus(self, embedder):
        # Three chunks with department metadata for RBAC filter test.
        texts = [
            ("life-1", "Life insurance provides death benefit and cash value.", "life"),
            ("home-1", "Home insurance covers dwelling, contents, and liability.", "home"),
            ("veh-1",  "Vehicle insurance covers collision, comprehensive, and liability.", "vehicle"),
        ]
        corpus = []
        for cid, text, dept in texts:
            emb = await embedder.embed_one(text)
            corpus.append(ChunkRecord(id=cid, content=text, embedding=emb, metadata={"dept": dept}))
        return corpus

    async def test_retrieves_relevant_chunk(self):
        e = FakeEmbedder(dim=256)
        corpus = await self._make_corpus(e)
        r = HybridRetriever(embedder=e)
        hits = await r.retrieve("life insurance death benefit", corpus, top_k_final=1)
        assert len(hits) == 1
        assert hits[0].id == "life-1"

    async def test_rbac_prefilter_hides_out_of_scope_chunks(self):
        """A vehicle-department employee must not see life or home chunks,
        even if the query semantically hits them."""
        e = FakeEmbedder(dim=256)
        corpus = await self._make_corpus(e)
        r = HybridRetriever(embedder=e)
        hits = await r.retrieve(
            "insurance coverage",
            corpus,
            top_k_final=5,
            filter_fn=lambda c: c.metadata.get("dept") == "vehicle",
        )
        for h in hits:
            assert h.metadata["dept"] == "vehicle"

    async def test_empty_corpus_returns_empty(self):
        r = HybridRetriever(embedder=FakeEmbedder())
        hits = await r.retrieve("anything", [], top_k_final=5)
        assert hits == []

    async def test_rbac_prefilter_leaving_no_candidates_returns_empty(self):
        e = FakeEmbedder(dim=256)
        corpus = await self._make_corpus(e)
        r = HybridRetriever(embedder=e)
        hits = await r.retrieve(
            "life insurance",
            corpus,
            filter_fn=lambda c: c.metadata.get("dept") == "nonexistent",
        )
        assert hits == []


# --- Reranker ---

@pytest.mark.asyncio
class TestReranker:
    async def test_lexical_reranker_prefers_higher_overlap(self):
        r = LexicalReranker()
        chunks = [
            RetrievedChunk(id="a", content="pizza dough recipe", score=0.9),
            RetrievedChunk(id="b", content="life insurance premium calculation", score=0.5),
        ]
        # Query has more overlap with b, so b should win despite lower initial score.
        out = await r.rerank("life insurance premium", chunks, top_k=1)
        assert out[0].id == "b"
