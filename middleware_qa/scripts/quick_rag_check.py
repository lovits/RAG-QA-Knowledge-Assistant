import json
import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, repo_root)
from middleware_qa.config.env import get_settings
from middleware_qa.agents import retrieval as R

def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "如何安装TongWeb"
    s = get_settings()
    print("env_backend=", s.RAG_BACKEND)
    print("topk=", s.RAG_TOPK, "min_sim=", getattr(s, "RAG_MIN_SIM", 0.65))
    items = R._query_chroma(q)
    print("filtered_count=", len(items))
    print("filtered_samples=", json.dumps(items[:2], ensure_ascii=False))
    try:
        import chromadb
        client = chromadb.PersistentClient(path=s.CHROMA_DIR)
        ef = R._EmbeddingFn(getattr(s, "EMBEDDING_MODEL", None))
        try:
            client.delete_collection(s.CHROMA_COLLECTION)
        except Exception:
            pass
        coll = client.get_or_create_collection(name=s.CHROMA_COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"})
        R._sync_chroma_from_resource(coll)
        r = coll.query(query_texts=[q], n_results=s.RAG_TOPK, include=["documents", "distances", "metadatas"])
        docs = (r.get("documents") or [[]])[0]
        dists = (r.get("distances") or [[]])[0]
        mets = (r.get("metadatas") or [[]])[0]
        print("raw_docs_count=", len(docs))
        print("raw_dists=", json.dumps(dists))
        print("raw_metas=", json.dumps(mets[:2], ensure_ascii=False))
    except Exception as e:
        print("raw_query_error=", str(e))

if __name__ == "__main__":
    main()
