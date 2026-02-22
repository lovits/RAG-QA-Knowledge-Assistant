from ..state_types import QASystemState

def make_security_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        config_snippet = state.get("config_snippet", "")
        answer = state.get("answer", "")
        prompt = f"""
You are a security auditor. Audit the configuration snippet and answer content for risks and provide fix suggestions.

Input:
- Config Snippet: {config_snippet}
- Answer Content: {answer}

Output Requirements (Must be standard JSON):
 {{
   "level": "High|Medium|Low",
   "risks": ["Max 3 risk points"],
   "fixes": ["Max 3 fix suggestions"]
 }}

Specs:
- Output ONLY one JSON object, no extra text, no code blocks, no comments.
- Use double quotes, do not use single quotes.
"""
        report = client.call_model(prompt)
        return {"security_report": report}
    return fn
