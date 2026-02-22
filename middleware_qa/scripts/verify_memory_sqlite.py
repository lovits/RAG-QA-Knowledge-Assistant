import os
import importlib.util as iu
from middleware_qa.llm.deepseek_client import DeepSeekClient
from middleware_qa.graph.workflow import compile_workflow, build_workflow

os.environ["CHECKPOINT_PATH"] = ".data/sqlite-verify.sqlite"
c = DeepSeekClient()

def _fast(prompt: str, retries: int = 2, backoff: float = 0.8) -> str:
    return "测试回答"

c.call_model = _fast
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    saver = SqliteSaver(path=os.environ["CHECKPOINT_PATH"])
    agent = build_workflow(c).compile(checkpointer=saver)
except Exception:
    agent = compile_workflow(c)
tid = "conv-sqlite-test"
q1 = "给我一个 Nginx 反向代理示例"
r1 = agent.invoke({"user_query": q1}, config={"configurable": {"thread_id": tid, "checkpoint_id": "step-1"}, "thread_id": tid})
history = [{"role": "user", "content": q1}, {"role": "assistant", "content": r1.get("answer", "")}]
q2 = "上面的配置怎么优化？"
r2 = agent.invoke({"user_query": q2, "history": history}, config={"configurable": {"thread_id": tid, "checkpoint_id": "step-2"}, "thread_id": tid})
print("summary_injected=", "历史摘要：" in r2.get("user_query", ""))
print("sqlite_module=", iu.find_spec("langgraph.checkpoint.sqlite") is not None)
p = os.environ["CHECKPOINT_PATH"]
print("sqlite_path=", p)
import os as _os
print("sqlite_exists=", _os.path.exists(p))
