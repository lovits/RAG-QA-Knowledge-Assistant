import logging
import re
from typing import Iterable

# 适配不同版本的pdfminer.six，解决extract_pages导入问题
try:
    # 高版本（20220524+）：extract_pages在high_level
    from pdfminer.high_level import extract_text as _fallback_extract, extract_pages
except ImportError:
    # 低版本：extract_pages在pdfpage，extract_text在high_level
    from pdfminer.high_level import extract_text as _fallback_extract
    from pdfminer.pdfpage import extract_pages

try:
    from pdfminer.layout import LAParams
except ImportError:
    # 极端低版本兼容
    from pdfminer.layout import LAParams as _LAParams
    LAParams = _LAParams

def _is_chinese(char: str) -> bool:
    """判断字符是否为中文字符"""
    return '\u4e00' <= char <= '\u9fff'

def _advanced_clean(text: str) -> str:
    """高级文本清洗：去除目录点线、页码、合并断行"""
    if not text:
        return ""
    
    # 1. 去除目录页常见的连续点号或下划线 (如 "Chapter 1 .......... 5")
    # 匹配连续4个及以上的点、下划线、破折号
    text = re.sub(r'[\._\-]{4,}', ' ', text)
    
    # 2. 去除单独占一行的纯数字 (通常是页码)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 3. 按行分割，准备合并
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return ""
        
    merged_lines = []
    buffer_line = ""
    
    for line in lines:
        # 过滤极短的无意义行 (如 "V8.0")，除非包含中文
        if len(line) < 3 and not any(_is_chinese(c) for c in line):
            continue
            
        if not buffer_line:
            buffer_line = line
            continue
            
        # 智能合并断行逻辑
        # 获取上一行的最后一个字符和当前行的第一个字符
        last_char = buffer_line[-1]
        first_char = line[0]
        
        # 情况A: 中文接中文 -> 直接合并，不加空格
        if _is_chinese(last_char) and _is_chinese(first_char):
            buffer_line += line
            
        # 情况B: 英文单词被连字符截断 (e.g. "install-", "ation") -> 去掉连字符合并
        elif last_char == '-' and not _is_chinese(first_char):
            buffer_line = buffer_line[:-1] + line
            
        # 情况C: 英文接英文/数字 -> 加空格合并 (如果是句子结束则不强行合并，视情况而定)
        # 这里简化处理：如果是句号结尾，说明是段落结束，推入 buffer 并开始新行
        elif last_char in '.!?。！？':
            merged_lines.append(buffer_line)
            buffer_line = line
        else:
            # 其他情况（如逗号结尾，或者普通断行），加空格合并（针对英文），中文环境其实不加空格也行
            # 为了通用性，如果两边都是非中文，加空格；有一边是中文，不加空格
            if not _is_chinese(last_char) and not _is_chinese(first_char):
                buffer_line += " " + line
            else:
                buffer_line += line
                
    if buffer_line:
        merged_lines.append(buffer_line)
        
    return "\n".join(merged_lines)

def pdf_extract_text(path: str, page_numbers: Iterable[int] | None = None, max_pages: int = 0) -> str:
    """修复+清洗+版本兼容版：提取PDF有效文本，过滤大量空白"""
    # 配置日志
    try:
        # 注意：日志输出的内容是自己在程序里面写的
        # 获得一个命名为pdf_extract的日志器
        logger = logging.getLogger("pdf_extract")
        # 给该日志器设置等级为INFO
        logger.setLevel(logging.INFO)
    except Exception:
        logger = logging.getLogger(__name__)

    # 第一步：优先完整提取文本（最稳定）
    try:
        # 页数有限制的话就提取对应的页数反之则提取整个文本
        if page_numbers is not None:
            raw_text = _fallback_extract(path, page_numbers=set(page_numbers), maxpages=max_pages) or ""
        else:
            raw_text = _fallback_extract(path, maxpages=max_pages) or ""
    except Exception as e:
        logger.error(f"完整提取失败：{path} | 错误：{str(e)[:200]}")
        raw_text = ""

    # 第二步：高级文本清洗
    if raw_text:
        clean_text = _advanced_clean(raw_text)
        logger.info(f"PDF清洗完成：{path} | 原始长度：{len(raw_text)} | 清洗后长度：{len(clean_text)}")
    else:
        clean_text = ""

    # 第三步：清洗后仍为空，尝试元素提取（备选方案）
    if not clean_text:
        try:
            # PDF 是分栏布局（；PDF 是图文混排（常规提取漏提被图片包围的文本块）；PDF 是表格 / 表单
            # 时清洗函数会失效采用LAParams()进行提取
            la = LAParams()
            # 保存提取的文本
            texts = []
            # 记录提取的页数
            count = 0
            # 提取pdf的每一页
            for page_layout in extract_pages(path, page_numbers=page_numbers, laparams=la):
                count += 1
                for element in page_layout:
                    # 兼容所有文本元素，避免漏提
                    if hasattr(element, "get_text"):
                        elem_text = element.get_text().strip()
                        # 使用同样的高级清洗
                        cleaned_elem = _advanced_clean(elem_text)
                        if cleaned_elem:
                            texts.append(cleaned_elem)
                # 如果max_pages不为0或者none（不要求全部提取）并且提取的页数超过了限制的提取页数
                if max_pages and count >= max_pages:
                    break
            clean_text = "\n".join(texts)
            logger.info(f"元素提取+清洗后：{path} | 长度：{len(clean_text)}")
        except Exception as e:
            logger.error(f"元素提取失败：{path} | 错误：{str(e)[:200]}")

    return clean_text