
import sys
import os
sys.path.append(os.getcwd())

from middleware_qa.utils.pdf_text import pdf_extract_text

pdf_path = r"c:\Users\79753\OneDrive\Desktop\industry\RagResource\TONG_data\TongWeb8\002_TongWeb_V8.0安装与使用指引.pdf"

if os.path.exists(pdf_path):
    print(f"Reading {pdf_path}...")
    try:
        text = pdf_extract_text(pdf_path, max_pages=3)
        print(f"Text length: {len(text)}")
        print(f"First 500 chars:\n{text[:500]}")
        if "安装" in text:
            print("✅ Contains '安装'")
        else:
            print("❌ Does NOT contain '安装'")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {pdf_path}")
