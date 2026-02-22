import sys
import os

# 将项目根目录添加到 python path，这样 Python 才知道如何解析 'middleware_qa' 包
sys.path.append(os.getcwd())

from middleware_qa.utils.scan_resource_files import scan_resource_files

if __name__ == "__main__":
    print("开始调试扫描...")
    # 这里调用函数，你可以把断点打在 scan_resource_files.py 内部
    results = scan_resource_files()
    print(f"扫描结束，找到 {len(results)} 个文件")
    for r in results[:3]:
        print(f"示例文件: {r[0]}")
