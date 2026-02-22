import re
from ..config.env import get_settings

# 清洗文本
def clean_text(t: str) -> str:
    # 读取配置
    s = get_settings()
    # 把\r\n替换成\n(windows) 同时把\r替换为\n(linux)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # 根据配置的正则表达式清洗文本，默认是移除所有空白字符
    pat = getattr(s, "RAG_CLEAN_REGEX", "")
    if pat:
        try:
            # 在t文本中将pat中的任务替换为空并flags=re.MULTILINE开启多行模式 
            t = re.sub(pat, "", t, flags=re.MULTILINE)
        except Exception:
            pass
    # 压缩水平空白
    t = re.sub(r"[ \t]+", " ", t)
    # 压缩垂直空白，将3个以上的换行替换为2个换行
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

