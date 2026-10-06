"""Hybrid retriever: FAISS dense search + BM25 keyword search, fused with RRF."""
import json
import re
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langsmith import traceable

from . import config


_STOP = set("a an the of to in on for and or is are was were be it its at by with as from that this "
            "these those what when how which who do does should can i my me you your there their".split())


def _tok(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


def expand_query(q: str) -> str:
    """Add agronomy vocabulary to broad questions (no chemical terms)."""
    ql = q.lower()
    expansions = []
    
    if re.search(r"how (to|do i|can i) (grow|cultivate|plant)|cultivation|package of practices", ql):
        expansions.append("climate soil land preparation sowing time seed rate spacing irrigation weeding harvesting")
    if "irrigat" in ql or "water" in ql:
        expansions.append("water management irrigation schedule critical stages drainage waterlogging")
    if "harvest" in ql:
        expansions.append("maturity indicators harvesting threshing yield drying storage moisture")
    if "storage" in ql or "store" in ql:
        expansions.append("post harvest drying moisture content storage moisture ventilation")
    if "pest" in ql or "disease" in ql:
        expansions.append("symptoms identification cultural control resistant varieties crop rotation preventive management")
    if "soil" in ql:
        expansions.append("soil fertility pH texture drainage water holding capacity organic matter")
    if "rain" in ql or "climate" in ql or "weather" in ql:
        expansions.append("rainfall temperature monsoon frost drought growing season")
    
    if expansions:
        return q + " " + " ".join(expansions)
    return q


class Retriever:
    def __init__(self):
        vs = config.VECTORSTORE_DIR
        if not (vs / "index.faiss").exists():
            raise FileNotFoundError("Vectorstore not found. Run:  python -m src.ingest")
        self.meta = json.loads((vs / "meta.json").read_text(encoding="utf-8"))
        if self.meta.get("domain") != "agriculture":
            raise RuntimeError("Vectorstore is not an agriculture index. Re-run: python -m src.ingest")
        self.chunks: List[Dict] = json.loads((vs / "chunks.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(vs / "index.faiss"))
        self.model = SentenceTransformer(self.meta["model"])
        self.bm25 = BM25Okapi([_tok(c["text"]) for c in self.chunks])

    @traceable(name="hybrid_retrieval", run_type="retriever", project_name=config.LANGSMITH_PROJECT)
    def search(self, query: str, k: int = config.TOP_K, crop: Optional[str] = None) -> List[Dict]:
        q = expand_query(query)
        qv = self.model.encode([q], normalize_embeddings=True).astype("float32")
        pool = len(self.chunks) if crop else min(len(self.chunks), 40)
        _, dense_ids = self.index.search(qv, pool)
        dense_ids = [int(i) for i in dense_ids[0] if i >= 0]
        bm = self.bm25.get_scores(_tok(q))
        bm_ids = [int(i) for i in np.argsort(bm)[::-1][:pool]]

        fused: Dict[int, float] = {}
        for ids in (dense_ids, bm_ids):
            for rank, i in enumerate(ids):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)

        results = []
        for i in sorted(fused, key=fused.get, reverse=True):
            c = self.chunks[i]
            if crop and c["crop"] != crop:
                continue
            score = float(np.dot(qv[0], self.index.reconstruct(i)))  # cosine
            results.append({**c, "score": score})
            if len(results) >= k:
                break
        return results
