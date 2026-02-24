**RAG‑QA 知识助手**（Retrieval‑Augmented Generation 问答系统）的演示 / 原型项目，结构清晰、功能全面，适合作为面试展示或快速搭建知识检索服务的参考。

---

## 🔍 项目概述

- **目标**  
  利用文档检索 + 大语言模型创建一个可以上传资料、构建向量数据库并对用户提问进行智能回答的服务。  
- **架构**  
  FastAPI 提供 REST 接口和简易前端；核心逻辑封装在 middleware_qa 包中，支持多种代理(agent)工作流用于不同任务（配置转换、故障排查、安全评估等）。

---

## ✅ 主要功能

1. **用户管理**  
   SQLite 后端，注册/登录、资料更新、连续失败锁定账号。

2. **文档摄取**  
   支持 PDF、TXT、Markdown、DOCX、CSV/Excel 等格式。文件清洗、分块后用 HuggingFace 嵌入模型生成向量，存入 ChromaDB。

3. **RAG 聊天接口**  
   `/api/chat` 根据检索到的文档上下文和 DeepSeek LLM 产生回答，支持流式返回与用户历史、个性化。

4. **分析工具**  
   文档摘要、跨文件比较、自动表格解析并导出 Excel 等。

5. **文件管理**  
   列表、单个删除、清空整个数据集/数据库。

6. **中间件 QA 工作流**  
   middleware_qa 中的模块构成可组合代理图，按意图路由检索、分析、生成，实现灵活的任务流水线。

---

## 🛠 技术栈

- **语言 & 框架**  
  Python 3.10+，FastAPI（Web 服务）、UNIX shell 脚本（启动）。

- **数据库**  
  SQLite（用户、会话等）。

- **向量检索**  
  ChromaDB 存储向量，HuggingFace sentence‑transformers 生成嵌入。

- **LLM / API**  
  DeepSeek 客户端 (deepseek_client.py) 做为生成模型后端。

- **依赖库**  
  `langchain`、`pandas`、`passlib`、`python-multipart` 等常见 AI/数据处理库。

- **项目结构亮点**  
  - backend：数据库模型、RAG 工具。  
  - middleware_qa：代理、意图、工作流、实用函数。  
  - static：网页前端。  
  - tests：基础单元测试。  
  - 脚本目录 (scripts) 用于快速检查与评估。

---

## 🗂 目录结构

```
main.py               # FastAPI 启动
backend/              # 用户与 RAG 辅助
middleware_qa/        # 代理与工作流
  … agents、api、graph、llm、registry、utils 等
static/               # 前端 HTML
tests/                # 单元测试
debug_*.py            # 调试脚本
```

---

## 📌 使用流程（概览）

1. 安装依赖：`pip install -r requirements.txt`  
2. 配置环境（`.env` 设置向量库目录、模型、API key 等）。  
3. 启动服务：`python main.py`，访问 `http://localhost:8000`。  
4. 上传/摄取文档 → 建立向量库 → 通过 `/api/chat` 提问或使用前端交互。

---

### 💡 面试讨论点

- RAG 架构与文档加载、检索上下文组装
- Chat 个性化与用户画像追踪
- FastAPI 的流式响应、文件处理、安全设置
- 代理工作流如何通过 `langgraph` 组合并扩展
- 可能的改进：安全性、容错、水平扩展、多模型支持等

---

该项目为简洁可扩展的知识问答平台示例，涵盖从文档摄取到智能回复的全链路，技术栈现代且具有演示价值。
