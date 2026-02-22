import os
import time
import json
from middleware_qa.llm.deepseek_client import DeepSeekClient
from middleware_qa.graph.workflow import compile_workflow
from middleware_qa.config.env import get_settings

def main():
    s = get_settings()
    d = getattr(s, "EVAL_DIR", ".obs") or ".obs"
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    path = os.path.join(d, "eval_results.jsonl")
    c = DeepSeekClient()
    agent = compile_workflow(c)
    tid = "eval-001"
    tests = [
        {"user_query": "给我一个 Nginx 反向代理示例", "conversation_id": tid},
        {"user_query": "上面的配置怎么优化？", "conversation_id": tid},
    ]
    for i, t in enumerate(tests):
        st = time.time()
        r = agent.invoke({"user_query": t["user_query"], "history": []}, {"configurable": {"thread_id": t["conversation_id"]}})
        ed = time.time()
        rec = {
            "ts": int(ed),
            "idx": i,
            "latency_ms": int((ed - st) * 1000),
            "intent": r.get("intent", ""),
            "answer_len": len(r.get("answer", "") or ""),
            "has_summary": ("历史摘要：" in str(r.get("user_query", ""))),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
