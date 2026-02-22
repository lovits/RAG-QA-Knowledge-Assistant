from sentence_transformers import SentenceTransformer

model_path = r"C:\Users\79753\OneDrive\Desktop\industry\.hf_cache\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
try:
    model = SentenceTransformer(model_path)
    print("✅ 模型加载成功！")
    test_embedding = model.encode("如何安装tongweb")
    print("向量维度:", len(test_embedding))
except Exception as e:
    print("❌ 模型加载失败:", e)
