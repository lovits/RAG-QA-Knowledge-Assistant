"""RAG Retrieval Augmented Agent

Responsibilities:
- Inject relevant background fragments for user questions before entering Pure QA.
- Default to using local persistent Chroma as vector retrieval backend; fallback to text/Markdown/PDF in resource directory on failure.

Config Sources:
- `RAG_BACKEND`, `CHROMA_DIR`, `CHROMA_COLLECTION`, `RAG_TOPK`, `RAG_MERGE_N`, `RAG_RESOURCE_DIR`, `EMBEDDING_MODEL`
"""
from ..state_types import QASystemState
# RAG Agent: Provides context background for user questions before entering Pure QA.
import os
import hashlib
from ..config.env import get_settings
from ..utils.resolve_model_path import resolve_model_path as _resolve_model_path
from ..utils.clean_text import clean_text as _clean_text
from ..utils.chunk_text import chunk_text as _chunk_text
from ..utils.scan_resource_files import scan_resource_files as _scan_resource_files
from ..logger import get_logger

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as _ChromaSTEF
except Exception:
    _ChromaSTEF = None


_s = get_settings()              # Read .env config (RAG_BACKEND, CHROMA_DIR, CHROMA_COLLECTION etc.)
_backend = _s.RAG_BACKEND        # Backend selection: chroma or none (fallback)
_logger = get_logger("RetrievalAgent")  # Enable logging

try:
    if getattr(_s, "HF_HOME", ""):  # Read Hugging Face related config items and set as system env vars
        os.environ["HF_HOME"] = _s.HF_HOME
    if getattr(_s, "HF_OFFLINE", ""):  # When HF_OFFLINE is "1", enable offline mode (use local models only)
        os.environ["TRANSFORMERS_OFFLINE"] = _s.HF_OFFLINE
except Exception:
    pass

class _EmbeddingFn:
    """
    Simple Embedding Function Wrapper.
    - Prioritize using `SentenceTransformer` to generate normalized vectors.
    - Downgrade to character-based fixed-dimension hash features when unavailable or failed.
    Args:
      - model_name: Specify embedding model name, default is `all-MiniLM-L6-v2`.
    Attributes:
      - dim: Dimension of fallback vectors (default 256).
    """
    def __init__(self, model_name: str | None = None):
        self.model = None
        self.model_name = model_name
        if SentenceTransformer is not None:
            try:
                # Store download location
                # Using cachekw = {} is defensive programming
                # Because passing "cache_folder": _s.HF_HOME might cause crashes for some models if None is passed
                cachekw = {}
                try:
                    if getattr(_s, "HF_HOME", ""):
                        cachekw = {"cache_folder": _s.HF_HOME}
                except Exception:
                    cachekw = {}
                
                # Store local model path
                mp = _resolve_model_path(model_name or "all-MiniLM-L6-v2")
                # **cachekw is equivalent to cache_folder="/my/path"
                self.model = SentenceTransformer(mp or (model_name or "all-MiniLM-L6-v2"), **cachekw)
                print(f"[EmbeddingFn] Successfully loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                print(f"[EmbeddingFn] Failed to load model, using Dummy Fallback. Error: {e}")
                self.model = None
        else:
            print("[EmbeddingFn] SentenceTransformer library not found, using Dummy Fallback.")

        # "Spare Tire" (Fallback) parameter, only effective when model loading fails.
        self.dim = 256
        
    # Internal use for Chroma database
    # In your business logic, you will never actively call ef.name(). This is purely to satisfy Chroma interface specs (Protocol).
    def name(self):
        return "EmbeddingFn"
    
    def __call__(self, input):
        target = input
        # Check if it is string type
        # How to ensure passing other types of parameters won't error (e.g. dict)?
        # In chroma:
        # def add(self, documents, ...):
        #     1. Check if documents is a list
        #     if documents is not None:
        #         if not isinstance(documents, list):
        #         raise ValueError("Documents must be a list")
        # So chroma internally checks if it is List[str] type
        if isinstance(input, str):
            target = [input]

        # Best method using SentenceTransformer
        if self.model is not None:
            try:
                # normalize_embeddings=True enables vector normalization
                # tolist() converts to Python list after normalization
                return self.model.encode(target, normalize_embeddings=True).tolist()
            except Exception as e:
                print(f"[EmbeddingFn] Model encoding failed: {e}")
                pass
        
        # Use custom simple encoding (meaningless)
        def enc(t: str):
            if not isinstance(t, str):
                t = str(t)
            v = [0.0] * self.dim
            for i, ch in enumerate(t):
                v[i % self.dim] += (ord(ch) % 13) / 13.0
            return v
        return [enc(d) for d in target]

    # Interface reserved for compatibility with other frameworks (like LangChain) or specific versions of Chroma protocol.
    def embed_query(self, input):
        res = self.__call__(input)
        if isinstance(input, list):
             return res
        return res[0] if res else []

    # Interface reserved for compatibility with other frameworks (like LangChain) or specific versions of Chroma protocol.
    def embed_documents(self, input):
        return self.__call__(input)


# Chroma Client: Equivalent to the entire Excel file (database connection).
# Collection (coll): Equivalent to one of the worksheets, specifically for storing a type of data.
# coll = client.get_or_create_collection(
#     name=_s.CHROMA_COLLECTION,  # e.g. "rag_docs"
#     ...
# )
# Equivalent to telling the database: "Get me the table named rag_docs, I want to save data into it, or query data from it."
def _sync_chroma_from_resource(coll):
    """
    Sync files scanned from resource directory with Chroma collection:
    - Add missing documents
    - Update documents with changed content/path/timestamp
    - Delete documents removed from resource directory
    Metadata fields: `path` `mtime` `sha256`
    """
    files = _scan_resource_files()
    if not files:
        return
    try:
         # Get metadatas from specific table (custom fields below)
         # Here existing is a dict, containing ids and metadatas, not just metadatas
        existing = coll.get(include=["metadatas"]) or {}
        # Extract metadatas and ids (here taking values from existing, not files)
        mets = existing.get("metadatas") or []
        # Here ids are cid
        ids_list = existing.get("ids") or []
        existing_by_id = {}
        # Merge two parallel lists (mets and ids_list) into a dict, key is ids, value is metadatas
        for i in range(min(len(ids_list), len(mets))):
            existing_by_id[ids_list[i]] = mets[i] or {}
        to_add_docs: list[str] = []
        to_add_ids: list[str] = []
        to_add_metas: list[dict] = []
        to_up_docs: list[str] = []
        to_up_ids: list[str] = []
        to_up_metas: list[dict] = []
        new_ids = set()
        for rid, content, rel, mt, sha in files:
            # Replace \\ with /
            reln = rel.replace("\\", "/")
            # Clean text
            cleaned = _clean_text(content)
            # Chunk text            
            chunks = _chunk_text(cleaned)
            # Stored in vector db as chunks
            for idx, ch in enumerate(chunks):
                # Example: 'file::TONG_data/THS6/001_TongHttpServer_V6.0_UserManual_6015A01.pdf#c00000'
                cid = f"{rid}#c{idx:05d}"
                # Example: '5c90d187092faec864664f8fe52790859ce1cb097ed00ee1c37f456f2e6d7796'
                csha = hashlib.sha256(ch.encode("utf-8", errors="ignore")).hexdigest()
                new_ids.add(cid)
                # 'sha256' ='5c90d187092faec864664f8fe52790859ce1cb097ed00ee1c37f456f2e6d7796'
                # 'mtime' =1760085883.079
                # 'path' ='TONG_data/THS6/001_TongHttpServer_V6.0_UserManual_6015A01.pdf'
                # 'chunk_index' =0
                # len() =4
                # Check if it exists in coll
                m = existing_by_id.get(cid)
                # If None, it should be added
                if m is None:
                    to_add_docs.append(ch)
                    to_add_ids.append(cid)
                    to_add_metas.append({"path": reln, "mtime": mt, "sha256": csha, "chunk_index": idx})
                
                else:
                    # If not None, check if update is needed
                    old_sha = m.get("sha256")
                    old_mtime = m.get("mtime")
                    old_path = (m.get("path") or "").replace("\\", "/")
                    if old_sha != csha or old_mtime != mt or old_path != reln:
                        to_up_docs.append(ch)
                        to_up_ids.append(cid)
                        to_up_metas.append({"path": reln, "mtime": mt, "sha256": csha, "chunk_index": idx})
        batch_size = 2000
        # Note coll.add and coll.update will call _EmbeddingFn in background because text needs to be stored, but coll.delete won't
        # If new documents
        if to_add_docs:
            # 0: start, len(to_add_docs) end, batch_size step. Example: 0-2000 2000-4000 [start:end)
            for i in range(0, len(to_add_docs), batch_size):
                # Add chunked docs from i to i+batch_size to coll
                end = i + batch_size
                coll.add(documents=to_add_docs[i:end], ids=to_add_ids[i:end], metadatas=to_add_metas[i:end])
        if to_up_docs:
            # "Repair if possible, replace if not, otherwise forget it."
            for i in range(0, len(to_up_docs), batch_size):
                end = i + batch_size
                # Best case: update directly
                try:
                    coll.update(ids=to_up_ids[i:end], documents=to_up_docs[i:end], metadatas=to_up_metas[i:end])
                # Secondary case: delete old then add updated content
                except Exception:
                    try:
                        # ids in coll are special keys, deleting ids removes related content
                        coll.delete(ids=to_up_ids[i:end])
                        coll.add(documents=to_up_docs[i:end], ids=to_up_ids[i:end], metadatas=to_up_metas[i:end])
                    # Worst case: ignore
                    except Exception:
                        pass
        # Convert existing_by_id.keys() to set for operations
        existing_ids = set(existing_by_id.keys())
        # Delete old IDs not present in latest chunks (including old whole file IDs)
        # Calculate items in existing_ids but not in new_ids
        to_del_ids = list(existing_ids - new_ids)
        if to_del_ids:
            # Still runs if batch_size > len(to_del_ids)
            for i in range(0, len(to_del_ids), batch_size):
                end = i + batch_size
                try:
                    coll.delete(ids=to_del_ids[i:end])
                except Exception:
                    pass
    except Exception as e:
        _logger.error(f"Sync chroma failed: {e}")
        return



def _query_chroma(q: str):
    """
    Use persistent `Chroma` collection for Top-K retrieval.
    - Collection name specified by `CHROMA_COLLECTION`.
    - Embedding function prioritizes `SentenceTransformerEmbeddingFunction`, otherwise degrades to `_EmbeddingFn`.
    - Calls `_sync_chroma_from_resource` before first query to ensure collection matches resource directory.
    Args:
      - q: Query string.
    Returns:
      - List of document contents (length not exceeding `RAG_TOPK`), returns empty list on failure.
    """
    # Connect to local persistent Chroma instance; attempt to import base corpus from RAG_RESOURCE_DIR if collection is empty.
    # Return list of most relevant document contents (max 3).
    if chromadb is None:
        return []
    try:
        # Pass chromadb path
        client = chromadb.PersistentClient(path=_s.CHROMA_DIR)
        # Initialize _EmbeddingFn
        ef = _EmbeddingFn(getattr(_s, "EMBEDDING_MODEL", None))
        # name=_s.CHROMA_COLLECTION refers to specific coll name (database has many tables)
        # Use 'cosine distance' to calculate similarity.
        coll = client.get_or_create_collection(
            name=_s.CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        _sync_chroma_from_resource(coll)
        # include=["documents", "distances", "metadatas"] specifies what info to include in return results.
        # Expand search range to support diversity filtering (MMR-like strategy)
        # Solve "too many docs from one version crowding out all results" problem
        # Since V7.0 has dozens of docs, we need a large enough net to catch V8.0
        # Set a large fixed minimum (200), or increase multiplier to 10x
        search_k = max(200, _s.RAG_TOPK * 10)
        r = coll.query(query_texts=[q], n_results=search_k, include=["documents", "distances", "metadatas"])
        docs = (r.get("documents") or [[]])[0]
        dists = (r.get("distances") or [[]])[0]
        mets = (r.get("metadatas") or [[]])[0]
        # Auto-fix abnormal data. If distance > 1.0, model might be wrong. Recreate collection.
        if dists and max([d for d in dists if d is not None], default=0.0) > 1.0:
            try:
                client.delete_collection(_s.CHROMA_COLLECTION)
            except Exception:
                pass
            coll = client.get_or_create_collection(
                name=_s.CHROMA_COLLECTION,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            _sync_chroma_from_resource(coll)
          
            r = coll.query(query_texts=[q], n_results=search_k, include=["documents", "distances", "metadatas"])
            docs = (r.get("documents") or [[]])[0]
            dists = (r.get("distances") or [[]])[0]
            mets = (r.get("metadatas") or [[]])[0]
        # Paranoia check
        if not docs or len(docs) != len(dists) or len(docs) != len(mets):
            return []
        
        # Distance -> Similarity. Closer distance means higher similarity.
        sims = [1 - (d if d is not None else 1.0) for d in dists]
        # Default threshold adjusted to 0.35 to adapt to weaker semantic matching models
        thr = getattr(_s, "RAG_MIN_SIM", 0.35)
        
        items = []
        path_counts = {}     # File level count
        group_counts = {}    # Directory/Group level count

        # [Diversity Control 1] Max recall chunks per file
        MAX_CHUNKS_PER_FILE = 2
        # [Diversity Control 2] Max recall chunks per similar directory (e.g. limit total for entire TongWEB7 series)
        MAX_CHUNKS_PER_GROUP = 4
        
        # Preprocessing: Pack and sort by similarity
        candidates = []
        for doc, sim, meta in zip(docs, sims, mets):
            candidates.append({"doc": doc, "sim": sim, "meta": meta})
        
        # Ensure descending sort by similarity
        candidates.sort(key=lambda x: x["sim"], reverse=True)
        
        _logger.info(f"Retrieved {len(candidates)} candidates from Chroma (search_k={search_k})")
        
        for cand in candidates:
            if cand["sim"] < thr:
                continue
            
            p = (cand["meta"] or {}).get("path", "")
            
            # --- Strategy 1: File level deduplication ---
            if path_counts.get(p, 0) >= MAX_CHUNKS_PER_FILE:
                continue

            # --- Strategy 2: Directory/Group level deduplication ---
            # Extract group feature: usually take first two directories in path as group name
            # e.g.: "TONG_data/TongWEB7/..." -> "TongWEB7"
            #       "TONG_data/TongWeb8/..." -> "TongWeb8"
            parts = p.replace("\\", "/").split("/")
            # parts:['TONG_data', 'TongWEB7', 'StandardEdition', 'InstallManual.pdf']
            # Try to extract distinctive directory name (assume 2nd level is version, e.g. TongWEB7, TongWeb8)
            # If path not long enough, use root directory
            group_key = parts[1] if len(parts) > 1 else (parts[0] if parts else "unknown")
            
            if group_counts.get(group_key, 0) >= MAX_CHUNKS_PER_GROUP:
                continue

            # Update counts
            path_counts[p] = path_counts.get(p, 0) + 1
            group_counts[group_key] = group_counts.get(group_key, 0) + 1
            
            # Get which chunk index this fragment is.
            # Although not deeply used in current logic,
            # if we implement "show context" later (e.g. show previous and next paragraph), this chunk_index is key coordinate.
            ci = (cand["meta"] or {}).get("chunk_index", 0)
            items.append({"doc": cand["doc"], "sim": float(cand["sim"]), "path": p, "chunk_index": ci})
            
            # Stop after filling RAG_TOPK
            if len(items) >= _s.RAG_TOPK:
                break
        
        return items
    except Exception as e:
        _logger.error(f"Query chroma failed: {e}")
        return []


def make_retrieval_agent(client):
    """
    Factory method: Create a Retrieval Augmented Agent node function.
    This function conforms to LangGraph node signature, reads `state["user_query"]`,
    and injects retrieved background fragments into returned `state` when available.
    Args:
      - client: Reserved parameter (currently unused)
    Returns:
      - Callable function `fn(state) -> state`
    """
    # Agent Factory: Returns function conforming to LangGraph node signature.
    def fn(state: QASystemState) -> QASystemState:
        """
        Enhance user question with background knowledge:
        - When backend is `chroma` and retrieval succeeds, merge top `RAG_MERGE_N` document fragments.
        - Inject max 1000 chars per fragment to control prompt length.
        - Record retrieval source and quantity in `intermediate_results`.
        """
        user_query = state["user_query"].strip()
        docs = []
        backend_used = "none"
        if _backend == "chroma":
            docs = _query_chroma(user_query)
            backend_used = "chroma" if docs else "none"
        # If no docs, do not inject background, treat as normal QA
        # Merge top RAG_MERGE_N retrieved doc fragments, control injection length (max 1000 chars/item)
        merged = "\n\n".join((it.get("doc", "")[:1000]) for it in docs[:_s.RAG_MERGE_N]) if docs else ""
        enhanced_query = (
            f"Answer based on the following retrieved background knowledge:\n{merged}\n\nOriginal User Question: {user_query}" if merged else user_query
        )
        # Record retrieval source and count, for runtime confirmation if Chroma was called
        _logger.info(f"backend={backend_used} results={len(docs)}")
        ir = state.get("intermediate_results", {})
        # Write retrieval source and count to intermediate results for subsequent nodes and external debugging
        ir_items = [{"path": it.get("path", ""), "sim": float(it.get("sim", 0.0))} for it in docs[:_s.RAG_MERGE_N]]
        # **ir: unpack all key-value pairs from old dict ir into here
        ir = {**ir, "retrieval_source": backend_used, "retrieval_docs_count": len(docs), "min_sim_threshold": getattr(_s, "RAG_MIN_SIM", 0.35), "retrieval_items": ir_items}
        return {**state, "user_query": enhanced_query, "intermediate_results": ir}
    return fn
