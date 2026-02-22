from ..state_types import QASystemState

def make_fault_troubleshoot_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        user_query = state.get("user_query", "")
        config_snippet = state.get("config_snippet", "")
        prompt = f"""
You are a middleware troubleshooting expert. Analyze the error info/logs and provide a solution.

Input:
- Question/Description: {user_query}
- Error Log/Config Snippet: {config_snippet}

Output Requirements:
- Analyze the root cause of the error;
- Provide specific solution steps;
- If info is insufficient, list the key info needed.
- Use numbered lists for steps.
"""
        try:
            troubleshooting_result = client.call_model(prompt)
            if not troubleshooting_result or troubleshooting_result.strip() == "":
                troubleshooting_result = "❌ Troubleshooting failed: Unable to identify the exception scenario, please provide more details and try again."
        except Exception:
            troubleshooting_result = "❌ Troubleshooting failed: Service busy, please try again later."
        return {"answer": troubleshooting_result}
    return fn
