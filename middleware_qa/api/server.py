import os
from ..llm.deepseek_client import DeepSeekClient
from ..graph.workflow import build_workflow, compile_workflow
from ..config.env import get_settings

def run_server(host: str = "0.0.0.0", port: int = 8000):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        import sqlite3
    except Exception as e:
        raise RuntimeError("fastapi and uvicorn are required to start the HTTP server")
    s = get_settings()
    client = DeepSeekClient()
    agent = compile_workflow(client)
    app = FastAPI()
    class Query(BaseModel):
        user_query: str
        conversation_id: str | None = None
    @app.post("/qa")
    def qa(q: Query):
        hist = []
        if q.conversation_id:
            try:
                conn = sqlite3.connect(s.MEMORY_DB_PATH)
                conn.execute("create table if not exists conversations(conversation_id text, role text, content text, ts integer)")
                cur = conn.execute("select role, content from conversations where conversation_id=? order by ts desc limit ?", (q.conversation_id, s.MEMORY_MAX_TURNS))
                rows = cur.fetchall()
                rows.reverse()
                hist = [{"role": r[0], "content": r[1]} for r in rows]
            except Exception:
                hist = []
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        cfg = {}
        if q.conversation_id:
            cfg = {"configurable": {"thread_id": q.conversation_id}}
        r = agent.invoke({"user_query": q.user_query, "history": hist}, cfg)
        ans = r.get("answer", "")
        if q.conversation_id:
            try:
                conn = sqlite3.connect(s.MEMORY_DB_PATH)
                conn.execute("create table if not exists conversations(conversation_id text, role text, content text, ts integer)")
                ts = int(__import__("time").time())
                conn.execute("insert into conversations(conversation_id, role, content, ts) values(?,?,?,?)", (q.conversation_id, "user", q.user_query, ts))
                conn.execute("insert into conversations(conversation_id, role, content, ts) values(?,?,?,?)", (q.conversation_id, "assistant", ans, ts+1))
                conn.commit()
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        ir = r.get("intermediate_results", {})
        retrieval = {
            "source": ir.get("retrieval_source", ""),
            "count": ir.get("retrieval_docs_count", 0),
            "min_sim": ir.get("min_sim_threshold", 0.0),
            "items": ir.get("retrieval_items", []),
        }
        return {"answer": ans, "intent": r.get("intent", ""), "retrieval": retrieval}
    uvicorn.run(app, host=host, port=port)
