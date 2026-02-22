import unittest
from middleware_qa.llm.deepseek_client import DeepSeekClient
from middleware_qa.graph.workflow import build_workflow

class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekClient()
        self.agent = build_workflow(self.client).compile()

    def test_pure_qa(self):
        r = self.agent.invoke({"user_query": "K8s 怎么查看某个命名空间下的Pod日志？"})
        self.assertTrue(len(r.get("answer", "")) > 0)

    def test_config_convert(self):
        q = "帮我把这个Nginx配置转换成Docker Compose：\nserver {\n    listen 80;\n    root /usr/share/nginx/html;\n}"
        r = self.agent.invoke({"user_query": q})
        self.assertTrue(len(r.get("answer", "")) > 0)

    def test_fault_troubleshoot(self):
        q = "K8s启动Pod报错：'Error: unable to connect to Redis at 192.168.1.100:6379: Connection refused'，帮我排查一下"
        r = self.agent.invoke({"user_query": q})
        self.assertTrue(len(r.get("answer", "")) > 0)

if __name__ == "__main__":
    unittest.main()
