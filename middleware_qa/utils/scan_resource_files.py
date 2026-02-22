import os
import hashlib
import pickle
import time
from ..config.env import get_settings
from ..logger import get_logger

_logger = get_logger("ResourceScanner")

try:
    from .pdf_text import pdf_extract_text as _pdf_extract_text
except Exception:
    _pdf_extract_text = None


def scan_resource_files():
    # 先找到资源文件夹的路径
    s = get_settings()  # 加载系统配置
    base = os.getcwd() # 获取当前工作目录
    # hasattr()检查一个对象是否包含指定名称的属性。
    resource_dir = os.path.join(base, s.RAG_RESOURCE_DIR if hasattr(s, "RAG_RESOURCE_DIR") else "RagResource")
    
    # 简单的文件缓存机制
    # .resource_cache.pkl保存的是mtime（最后修改时间）, content（文件内容）, sha256（文件内容的哈希值）的映射
    cache_path = os.path.join(base, ".resource_cache.pkl") 
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                # 加载序列化数据到内存
                cache = pickle.load(f)
        except Exception:
            cache = {}
    
    out = []
    exts = {".md", ".txt", ".yaml", ".yml", ".json", ".conf", ".ini", ".pdf"}
    cache_dirty = False
    
    # 记录本次扫描到的所有文件路径（为记录是否有文件删除做准备）
    scanned_paths = set()
    
    # 扫描文件夹下的所有文件
    if os.path.isdir(resource_dir):
        for root, _, files in os.walk(resource_dir):
            # 判断文件是否是可以被解析的，是的话就获取文件p的最后修改时间戳
            for name in files:
                _, ext = os.path.splitext(name)
                if ext.lower() in exts:
                    p = os.path.join(root, name) 
                    scanned_paths.add(p)
                    try:
                        mt = os.stat(p).st_mtime # 获取文件p的最后修改时间戳
                    except Exception:
                        continue

                    # 检查缓存
                    cached_entry = cache.get(p)
                    # 缓存格式: (mtime, content, sha)
                    
                    content = None
                    sha = None

                    # 如果缓存文件中的mtime与当前文件的mtime相同，说明文件未被修改，直接使用缓存
                    # 这里拿出缓存中的content, sha的目的是给下游使用，比如计算相似度时需要比较文件内容是否有变化
                    if cached_entry and cached_entry[0] == mt:
                        # 【新增检查】即使时间戳匹配，也要检查内容是否为空
                        # 如果内容是空的，说明之前的缓存是脏的（缓存中毒），必须强制忽略，重新解析
                        cached_content = cached_entry[1]
                        if cached_content and len(cached_content.strip()) > 0:
                            content, sha = cached_entry[1], cached_entry[2]

                    if content is None:
                        # 缓存未命中或过期，重新读取
                        try:
                            content = ""
                            # 如果判断是pdf文件并且_pdf_extract_text不为空
                            if ext.lower() == ".pdf" and _pdf_extract_text is not None:
                                try:
                                    # getattr=get attritube
                                    # 判断读取到pdf的第几页
                                    pages = int(getattr(s, "RAG_PDF_MAX_PAGES", 0) or 0)
                                    # 打印出要读取的页码范围 比如（0，1，2，3，4）
                                    nums = set(range(pages)) if pages > 0 else None
                                    # 获取pdf的内容
                                    content = _pdf_extract_text(p, page_numbers=nums, max_pages=pages) or ""
                                except Exception as e:
                                    _logger.warning(f"Failed to extract text from {name}: {e}")
                                    content = ""
                            else:
                                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read()
                        except Exception:
                            continue
                        
                        # 生成sha256
                        sha = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
                        
                        # 【优化】只有当成功提取到内容时才写入缓存
                        # 防止因为一次解析失败（content=""）导致"缓存中毒"，以后永远不再尝试解析
                        if content and len(content.strip()) > 0:
                            # 保存p（os.path.join(root, name)）的内容
                            cache[p] = (mt, content, sha)
                            # cache被修改过
                            cache_dirty = True
                        else:
                            # 如果内容为空，不存缓存，这样下次运行时还会尝试重新解析
                            pass

                    # 资源目录内的相对路径（用于定位文件）
                    # relpath：
                    # 'C:\\Users\\79753\\OneDrive\\Desktop\\industry\\RagResource\\TONG_data\\TongSearch_doc_3061\\013_TongSearch_V3.0数据迁移手册_3061A01.pdf'
                    # 减去
                    # 'C:\\Users\\79753\\OneDrive\\Desktop\\industry\\RagResource'
                    rel = os.path.relpath(p, resource_dir)
                    # 文件唯一标识（跨平台、全局唯一）
                    # os.sep代表各自系统的转义符 win系统是\\ linux是/               
                    # 将不论什么系统的分隔符都替换为/
                    # 如果直接用rel当标识符会出现问题如下
                    # Windows	pdf\人工智能.pdf	Linux	pdf/人工智能.pdf	
                    # 系统认为这是「新资源」，重复存储相同文本 + 向量
                    rid = "file::" + rel.replace(os.sep, "/")
                    out.append((rid, content, rel, mt, sha))

    # 清理缓存中已不存在的文件
    # python中不能在遍历字典（或集合）的同时修改它的大小 
    # 因此这里list相当于给当前的keys拍了一张快照
    for p in list(cache.keys()): 
        if p not in scanned_paths:
            del cache[p]
            cache_dirty = True
            # _logger.debug(f"Removed missing file from cache: {p}")

    # 如果cache被修改则尝试保存到cache_path
    if cache_dirty:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            _logger.warning(f"Failed to save resource cache: {e}")
            
    return out
