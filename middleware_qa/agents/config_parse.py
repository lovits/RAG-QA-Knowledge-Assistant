import json
from ..state_types import QASystemState

def make_config_parse_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        config_snippet = state["config_snippet"]
        prompt = f"""
You are a middleware configuration parsing expert. Process the configuration snippet strictly according to the following requirements:
Task:
1. Identify the configuration type (Must be one of: Docker Compose, Helm, K8s YAML, Nginx, Redis);
2. Parse the configuration snippet into structured JSON, clarifying the meaning of each field;
3. Retain core configuration, ignore comments and redundant fields.

Output Requirements:
- Output ONLY standard JSON, no code blocks, no comments, no extra text;
- Use double quotes for fields, must include two required fields: "config_type", "parsed_config".

Config Snippet: {config_snippet}
"""
        result_str = client.call_model(prompt)
        try:
            result_str = result_str.replace("'", '"')
            result_str = result_str.replace("```json", "").replace("```", "")
            result_str = result_str.strip()
            result_dict = json.loads(result_str)
            config_type = result_dict.get("config_type", "Unknown Config")
            parsed_config = result_dict.get("parsed_config", {})
            if not isinstance(parsed_config, dict) or len(parsed_config) == 0:
                parsed_config = {"error": "Config parse failed, no valid structured data extracted"}
        except json.JSONDecodeError:
            config_type = "Unknown Config"
            parsed_config = {"error": "Config format invalid, unable to parse"}
        except Exception:
            config_type = "Unknown Config"
            parsed_config = {"error": "Unknown error during config parse"}
        return {"config_type": config_type, "parsed_config": parsed_config}
    return fn
