from ..state_types import QASystemState

def make_supervisor_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        intent = state.get("intent", "Pure QA")
        user_query = state.get("user_query", "")
        prompt = f"""
You are the supervisor of a middleware QA system, responsible for setting goals and constraints for subsequent roles.

Input:
- Intent: {intent}
- Question: {user_query}

Output Requirements (Must be standard JSON):
 {{
   "goal": "One sentence summarizing the goal of this processing",
   "constraints": ["Length constraint", "Format constraint", "Risk constraint"]
 }}

Specs:
- Output ONLY one JSON object, no extra text, no code blocks, no comments.
- Use double quotes, do not use single quotes. Constraints max 3.
"""
        plan = client.call_model(prompt)
        return {"supervision": plan}
    return fn
