import json
from ..state_types import QASystemState
from ..registry.intents import allowed_intents

def make_intent_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        user_query = state["user_query"]
        allowed_list = allowed_intents()
        allowed_str = "/".join(allowed_list)
        description_str = "\n   " + "\n   ".join([f"- {i}" for i in allowed_list])
        prompt = f"""
You are the intent recognizer for a middleware QA system.

Task:
1. Choose one of the candidate intents: {allowed_str}
   Description:
{description_str}
2. Extract the configuration snippet or error log provided by the user (empty string "" if none).

Output Specification (Strictly Follow):
- Output ONLY a standard JSON object, no extra text or code blocks.
- Keys and values must use double quotes. No trailing commas.
- Fields must be exactly "intent" and "config_snippet".
- "intent" must be one of the candidate values (e.g., "Pure QA"), do NOT output the full list string.
- "config_snippet" is a string; use "" if empty; use \\n for newlines if multi-line.

Correct Example (No Snippet):
{{"intent": "Pure QA", "config_snippet": ""}}

Correct Example (With Snippet):
{{"intent": "Config Conversion", "config_snippet": "server {{\n  listen 80;\n  root /usr/share/nginx/html;\n}}"}}

Incorrect Example (Using list as value):
{{"intent": "{allowed_str}", "config_snippet": ""}}

Incorrect Example (Single quotes/Code block/Extra text):
{{'intent': 'Pure QA', 'config_snippet': ''}}
```json
{{"intent": "Pure QA", "config_snippet": ""}}
```
{{"intent": "Pure QA", "config_snippet": ""}} Please continue...

User Query: {user_query}
"""
        result_str = client.call_model(prompt)
        try:
            result_str = result_str.replace("'", '"').strip()
            result_dict = json.loads(result_str)
            intent = result_dict.get("intent", "Pure QA")
            config_snippet = result_dict.get("config_snippet", "")
            if intent not in allowed_list:
                intent = "Pure QA"
        except json.JSONDecodeError:
            intent = "Pure QA"
            config_snippet = ""
        return {"intent": intent, "config_snippet": config_snippet}
    return fn
