import os
import json
import time
import threading
from ..config.env import get_settings

class EventLogger:
    def __init__(self):
        s = get_settings()
        d = getattr(s, "OBS_DIR", ".obs") or ".obs"
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        self.path = os.path.join(d, "events.jsonl")
        self.lock = threading.Lock()

    def write(self, record: dict):
        try:
            line = json.dumps(record, ensure_ascii=False)
            with self.lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

def with_observability(node_name: str, fn):
    logger = EventLogger()
    def wrapper(state):
        t0 = time.time()
        try:
            logger.write({
                "ts": int(t0),
                "event": "start",
                "node": node_name,
                "user_query": str(state.get("user_query", ""))[:400],
                "history_count": len(state.get("history", []) or []),
            })
        except Exception:
            pass
        try:
            out = fn(state)
            t1 = time.time()
            try:
                logger.write({
                    "ts": int(t1),
                    "event": "end",
                    "node": node_name,
                    "latency_ms": int((t1 - t0) * 1000),
                    "answer_len": len(out.get("answer", "") or ""),
                    "intent": str(out.get("intent", "")),
                })
            except Exception:
                pass
            return out
        except Exception as e:
            t1 = time.time()
            try:
                logger.write({
                    "ts": int(t1),
                    "event": "error",
                    "node": node_name,
                    "latency_ms": int((t1 - t0) * 1000),
                    "error": repr(e),
                })
            except Exception:
                pass
            raise
    return wrapper
