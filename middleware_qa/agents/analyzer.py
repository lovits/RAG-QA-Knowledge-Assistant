from ..state_types import QASystemState

def make_analyzer_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        intent = state.get("intent", "Pure QA")
        user_query = state.get("user_query", "")
        config_snippet = state.get("config_snippet", "")
        prompt = f"""
You are an analyst. Extract structured information to support subsequent generation.

Input:
- Intent: {intent}
- Question: {user_query}
- Config Snippet: {config_snippet}

Output Requirements (Must be standard JSON):
 {{
   "key_points": ["Key information, max 3 points"],
   "content_type": "steps|commands|config_conversion|troubleshooting"
 }}

Specs:
- Output ONLY one JSON object, no extra text, no code blocks, no comments.
- Use double quotes, do not use single quotes.
"""
        analysis = client.call_model(prompt)
        return {"analysis": analysis}
    return fn
