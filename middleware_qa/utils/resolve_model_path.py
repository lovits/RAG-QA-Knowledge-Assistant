import os
from ..config.env import get_settings

def resolve_model_path(name_or_path: str | None) -> str | None:
    s = get_settings()
    p = (name_or_path or "").strip()
    if p and os.path.isdir(p):
        return p
    base = getattr(s, "HF_HOME", "")
    if not base or not os.path.isdir(base):
        return None
    hub = os.path.join(base, "hub")
    org = "sentence-transformers"
    model = (p or "all-MiniLM-L6-v2")
    rid1 = os.path.join(hub, f"models--{org}--{model}")
    rid2 = os.path.join(hub, "models--" + model.replace("/", "--"))
    for rid in (rid1, rid2):
        snaps = os.path.join(rid, "snapshots")
        if os.path.isdir(snaps):
            subs = [os.path.join(snaps, d) for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d))]
            if subs:
                subs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return subs[0]
    return None

