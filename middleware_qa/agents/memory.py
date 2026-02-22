from ..state_types import QASystemState

def make_memory_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        h = state.get("history", [])
        uq = state.get("user_query", "").strip()
        if isinstance(h, list) and len(h) > 0:
            xs = []
            for x in h[-6:]:
                try:
                    r = str(x.get("role", "")).strip()
                    c = str(x.get("content", "")).strip()
                    if r and c:
                        xs.append(f"{r}:{c}")
                except Exception:
                    pass
            hist = "\n".join(xs)
            if hist:
                try:
                    s = client.call_model(f"Based on the following multi-turn conversation history, generate a context summary of no more than 150 words. Output only the summary paragraph. History:\n{hist}\nCurrent Question: {uq}")
                    s = s.strip()
                    if s:
                        uq = f"History Summary: {s}\nCurrent Question: {uq}"
                except Exception:
                    pass
        return {**state, "user_query": uq}
    return fn
