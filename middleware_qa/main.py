import os
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .llm.deepseek_client import DeepSeekClient
from .graph.workflow import build_workflow, compile_workflow

def demo():
    client = DeepSeekClient()
    agent = compile_workflow(client)
    graph_png = agent.get_graph(xray=True).draw_mermaid_png()
    output_file = "agent_graph.png"
    with open(output_file, "wb") as f:
        f.write(graph_png)
    if 'get_ipython' in globals():
        from IPython.display import Image, display
        display(Image(graph_png))
    else:
        try:
            img_bytes = BytesIO(graph_png)
            img = mpimg.imread(img_bytes, format='png')
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('LangGraph Agent Workflow')
            plt.show()
        except Exception:
            pass
    user_query1 = "K8s 怎么查看某个命名空间下的Pod日志？"
    result1 = agent.invoke({"user_query": user_query1})
    print(result1["answer"])
    user_query2 = "帮我把这个Nginx配置转换成Docker Compose：\nserver {\n    listen 80;\n    root /usr/share/nginx/html;\n}"
    result2 = agent.invoke({"user_query": user_query2})
    print(result2["answer"])
    user_query3 = "K8s启动Pod报错：'Error: unable to connect to Redis at 192.168.1.100:6379: Connection refused'，帮我排查一下"
    result3 = agent.invoke({"user_query": user_query3})
    print(result3["answer"])

if __name__ == "__main__":
    demo()
