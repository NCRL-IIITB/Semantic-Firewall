"""Semantic memory: a vector cache of confirmed threats plus an admin allowlist.

Two ChromaDB collections (cosine HNSW) hold 384-d all-MiniLM-L6-v2 embeddings:

* ``threat_intel_cache`` - prompts confirmed as attacks. A query is blocked when its
  cosine similarity to the nearest entry is >= ``similarity_threshold`` (tau).
* ``semantic_allowlist`` - prompts an administrator approved after a false positive.

Write-back into the threat cache is policy-controlled (see ``add_threat``): by default
only LLM-confirmed detections with confidence >= ``writeback_min_confidence`` are stored,
and the oldest entries are evicted once ``max_entries`` is exceeded.
"""

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional

from semantic_firewall.core.orchestrator.paths import var_path

THREAT_COLLECTION = "threat_intel_cache"
ALLOWLIST_COLLECTION = "semantic_allowlist"

_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()


def shared_embedder():
    """Return one process-wide ONNX MiniLM-L6-v2 embedder.

    Chroma's default embedding function rebuilds its ONNX session on every query
    (~300 ms each); reusing a single instance brings an embedding down to ~30 ms.
    """
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

            _EMBEDDER = ONNXMiniLM_L6_V2()
        return _EMBEDDER


class SemanticCache:
    def __init__(
        self,
        db_path: Optional[str] = None,
        enabled: bool = True,
        similarity_threshold: float = 0.90,
        allowlist_threshold: float = 0.90,
        writeback: bool = True,
        writeback_require_llm: bool = True,
        writeback_min_confidence: float = 0.85,
        max_entries: int = 10000,
    ):
        self.db_path = db_path or str(var_path("chroma_db"))
        self.similarity_threshold = similarity_threshold
        self.allowlist_threshold = allowlist_threshold
        self.writeback = writeback
        self.writeback_require_llm = writeback_require_llm
        self.writeback_min_confidence = writeback_min_confidence
        self.max_entries = max_entries
        self._write_lock = threading.Lock()
        self.stats = {
            "threat_queries": 0,
            "threat_hits": 0,
            "allowlist_queries": 0,
            "allowlist_hits": 0,
            "writes": 0,
            "writes_rejected": 0,
            "evictions": 0,
        }
        self.enabled = False
        if not enabled:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            print(f"[SemanticCache] Initializing local Vector DB at {self.db_path}...")
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self._open_collections()
            self.embedder = shared_embedder()
            self.enabled = True
            print("[SemanticCache] Vector DB successfully initialized.")
        except Exception as e:
            print(f"[SemanticCache] Failed to initialize ChromaDB: {e}. Semantic Caching disabled.")
            self.enabled = False

    def _open_collections(self):
        self.collection = self.client.get_or_create_collection(
            name=THREAT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.allowlist_collection = self.client.get_or_create_collection(
            name=ALLOWLIST_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ── Embedding and lookup ─────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        return [float(x) for x in self.embedder([text])[0]]

    def embed_many(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            for vec in self.embedder(texts[start:start + batch_size]):
                vectors.append([float(x) for x in vec])
        return vectors

    def _nearest(self, collection, embedding: List[float]) -> Optional[Dict[str, Any]]:
        if collection.count() == 0:
            return None
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return None
        distance = float(results["distances"][0][0])
        return {
            "matched_text": results["documents"][0][0],
            "metadata": results["metadatas"][0][0] or {},
            "distance": distance,
            "similarity_score": 1.0 - distance,
        }

    def nearest_threat_similarity(self, text: str, embedding: Optional[List[float]] = None) -> Optional[float]:
        """Similarity to the closest cached threat, regardless of the threshold."""
        if not self.enabled:
            return None
        nearest = self._nearest(self.collection, embedding or self.embed(text))
        return None if nearest is None else nearest["similarity_score"]

    def lookup_threat(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> tuple[Optional[float], Optional[Dict[str, Any]]]:
        """One nearest-neighbour query: (similarity to nearest threat, hit if similarity >= tau)."""
        if not self.enabled:
            return None, None
        threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
        try:
            self.stats["threat_queries"] += 1
            nearest = self._nearest(self.collection, embedding or self.embed(text))
            if nearest is None:
                return None, None
            if nearest["similarity_score"] < threshold:
                return nearest["similarity_score"], None
            self.stats["threat_hits"] += 1
            metadata = nearest["metadata"]
            return nearest["similarity_score"], {
                "matched_text": nearest["matched_text"],
                "threat_type": metadata.get("threat_type", "INJECTION"),
                "severity": metadata.get("severity", "HIGH"),
                "source_agent": metadata.get("source_agent", "unknown"),
                "origin": metadata.get("origin", "online"),
                "distance": nearest["distance"],
                "similarity_score": nearest["similarity_score"],
            }
        except Exception as e:
            print(f"[SemanticCache] Error querying cache: {e}")
            return None, None

    def check_threat(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the nearest cached threat if its similarity is >= tau, else None."""
        return self.lookup_threat(text, embedding=embedding, similarity_threshold=similarity_threshold)[1]

    def check_allowlist(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        threshold = self.allowlist_threshold if similarity_threshold is None else similarity_threshold
        try:
            self.stats["allowlist_queries"] += 1
            nearest = self._nearest(self.allowlist_collection, embedding or self.embed(text))
            if nearest is None or nearest["similarity_score"] < threshold:
                return None
            self.stats["allowlist_hits"] += 1
            return {
                "matched_text": nearest["matched_text"],
                "reason": nearest["metadata"].get("reason", "unknown"),
                "distance": nearest["distance"],
                "similarity_score": nearest["similarity_score"],
            }
        except Exception as e:
            print(f"[SemanticCache] Error querying allowlist cache: {e}")
            return None

    # ── Writes ───────────────────────────────────────────────────────────────

    def writeback_allowed(self, confidence: float, llm_confirmed: bool) -> bool:
        if not self.writeback:
            return False
        if self.writeback_require_llm and not llm_confirmed:
            return False
        return confidence >= self.writeback_min_confidence

    def add_threat(
        self,
        text: str,
        threat_type: str,
        severity: str,
        source_agent: str,
        confidence: float = 1.0,
        llm_confirmed: bool = False,
        origin: str = "online",
        force: bool = False,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Store a detected threat. Returns True if a new entry was written.

        Online write-backs must pass ``writeback_allowed``; ``force=True`` is for
        explicit seeding (curated feeds, warm-cache experiments) and skips that check.
        """
        if not self.enabled:
            return False
        if not force and not self.writeback_allowed(confidence, llm_confirmed):
            self.stats["writes_rejected"] += 1
            return False

        try:
            doc_id = self._generate_id(text)
            with self._write_lock:
                existing = self.collection.get(ids=[doc_id])
                if existing and existing.get("ids"):
                    return False
                self.collection.add(
                    ids=[doc_id],
                    documents=[text],
                    embeddings=[embedding or self.embed(text)],
                    metadatas=[{
                        "threat_type": threat_type,
                        "severity": severity,
                        "source_agent": source_agent,
                        "confidence": float(confidence),
                        "llm_confirmed": bool(llm_confirmed),
                        "origin": origin,
                        "timestamp": time.time(),
                    }],
                )
                self.stats["writes"] += 1
                self._evict_locked()
            print(f"[SemanticCache] Added new threat signature to Vector DB (type: {threat_type}).")
            return True
        except Exception as e:
            print(f"[SemanticCache] Error adding threat to cache: {e}")
            return False

    def seed_threats(self, texts: List[str], threat_type: str = "INJECTION", severity: str = "HIGH",
                     origin: str = "seed") -> int:
        """Bulk-insert known attack prompts (used for the warm-cache protocol)."""
        if not self.enabled or not texts:
            return 0
        unique = list(dict.fromkeys(texts))
        ids = [self._generate_id(t) for t in unique]
        existing = set(self.collection.get(ids=ids).get("ids") or [])
        new = [(i, t) for i, t in zip(ids, unique) if i not in existing]
        if not new:
            return 0
        now = time.time()
        embeddings = self.embed_many([t for _, t in new])
        for start in range(0, len(new), 500):
            chunk = new[start:start + 500]
            self.collection.add(
                ids=[i for i, _ in chunk],
                documents=[t for _, t in chunk],
                embeddings=embeddings[start:start + len(chunk)],
                metadatas=[{
                    "threat_type": threat_type,
                    "severity": severity,
                    "source_agent": "seed",
                    "confidence": 1.0,
                    "llm_confirmed": False,
                    "origin": origin,
                    "timestamp": now,
                } for _ in chunk],
            )
        with self._write_lock:
            self._evict_locked()
        return len(new)

    def _evict_locked(self):
        overflow = self.collection.count() - self.max_entries
        if overflow <= 0:
            return
        records = self.collection.get(include=["metadatas"])
        ordered = sorted(
            zip(records["ids"], records["metadatas"]),
            key=lambda item: (item[1] or {}).get("timestamp", 0.0),
        )
        victims = [doc_id for doc_id, _ in ordered[:overflow]]
        self.collection.delete(ids=victims)
        self.stats["evictions"] += len(victims)

    def add_allowlist(self, text: str, reason: str = "manual_override"):
        """Add an administrator-approved prompt to the semantic allowlist."""
        if not self.enabled:
            return

        try:
            doc_id = self._generate_id(text)
            existing = self.allowlist_collection.get(ids=[doc_id])
            if existing and existing.get("ids"):
                return
            self.allowlist_collection.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[self.embed(text)],
                metadatas=[{"reason": reason, "timestamp": time.time()}],
            )
            print("[SemanticCache] Added new safe signature to Allowlist Vector DB.")
        except Exception as e:
            print(f"[SemanticCache] Error adding allowlist to cache: {e}")

    # ── Maintenance ──────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count() if self.enabled else 0

    def reset(self):
        """Drop both collections (used to start every experiment from an empty cache)."""
        if not self.enabled:
            return
        for name in (THREAT_COLLECTION, ALLOWLIST_COLLECTION):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        self._open_collections()
        for key in self.stats:
            self.stats[key] = 0
