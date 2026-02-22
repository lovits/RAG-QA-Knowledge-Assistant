
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

# Force RAG settings
os.environ["RAG_BACKEND"] = "chroma"
os.environ["RAG_MIN_SIM"] = "0.1" # Lower threshold to see all results

from middleware_qa.agents.retrieval import _query_chroma
from middleware_qa.utils.scan_resource_files import scan_resource_files
from middleware_qa.config.env import get_settings

def test_rag():
    print("Initializing RAG test...")
    
    # Debug: Check scanned files
    print("Scanning resource files...")
    files = scan_resource_files()
    print(f"Total files scanned: {len(files)}")
    
    install_guide_found = False
    # for rid, content, rel, mt, sha in files:
    #     if "安装" in rel and "TongWeb" in rel:
    #         print(f"Found Install Guide: {rel} (Content len: {len(content)})")
    #         install_guide_found = True
    #         # Print first 100 chars of content to verify it's not empty
    #         print(f"  Content preview: {content[:100].replace(chr(10), ' ')}...")

    if not install_guide_found:
        print("WARNING: No TongWeb installation guide found in scanned files!")
    
    # Trigger RAG initialization and sync
    print("\nTriggering RAG sync (this might take a moment)...")
    _query_chroma("warmup")

    # Inspect Chroma directly
    import chromadb
    client = chromadb.PersistentClient(path=get_settings().CHROMA_DIR)
    coll = client.get_collection(get_settings().CHROMA_COLLECTION)
    
    # Check for installation guide in DB
    target_path = "TONG_data/TongWeb8/002_TongWeb_V8.0安装与使用指引.pdf"
    print(f"\nChecking Chroma for: {target_path}")
    existing = coll.get(where={"path": target_path})
    if existing and existing['ids']:
        print(f"Found {len(existing['ids'])} chunks for this file in Chroma.")
    else:
        print("❌ File NOT found in Chroma DB!")
# 测试查询
    query = "如何安装tongnweb"  # 故意写错为 TongNWeb
    print(f"Querying: {query}")
    
    try:
        # Call the internal function directly
        results = _query_chroma(query)
        
        print("\n=== RAG Retrieval Results ===")
        if not results:
            print("No documents retrieved.")
        else:
            print(f"Found {len(results)} documents:")
            for i, item in enumerate(results):
                print(f"\n[Result {i+1}]")
                print(f"Path: {item.get('path')}")
                print(f"Similarity: {item.get('sim'):.4f}")
                print(f"Chunk Index: {item.get('chunk_index')}")
                # Print a preview of the content
                content = item.get('doc', '')
                preview = content[:200].replace('\n', ' ')
                print(f"Content Preview: {preview}...")
                
    except Exception as e:
        print(f"Error during RAG test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag()
