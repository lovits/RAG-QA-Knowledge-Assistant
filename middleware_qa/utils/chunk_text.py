from ..config.env import get_settings

# 文本分块
def chunk_text(t: str) -> list[str]:
    # 读取配置
    s = get_settings()
    try:
        # 读取文本的分块大小
        size = int(getattr(s, "RAG_CHUNK_SIZE", 512))
    except Exception:
        size = 512
    try:
        # 分块的文本叠加的大小 目的：防止语义被切刀切断 而设置的“缓冲区”
        overlap = int(getattr(s, "RAG_CHUNK_OVERLAP", 0))
    except Exception:
        overlap = 0
    if size <= 0:
        return [t]
    # 确定每次前进的步数
    step = size - overlap if overlap and overlap < size else size
    out: list[str] = []
    i = 0
    n = len(t)
    # 具体的分块操作
    while i < n:
        j = min(i + size, n)
        out.append(t[i:j])
        if j >= n:
            break
        i += step
    return out

