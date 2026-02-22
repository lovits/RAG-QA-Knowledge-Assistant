from ..state_types import QASystemState

def make_config_convert_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        parsed_config = state["parsed_config"]
        user_query = state["user_query"].strip()
        config_type = state.get("config_type", "Unknown Config")
        if isinstance(parsed_config, dict) and "error" in parsed_config:
            converted_config = f"Config conversion failed: Original config parse exception ({parsed_config['error']}), unable to convert."
            return {**state, "answer": converted_config}
        prompt = f"""
# Task Description
You are a middleware configuration conversion expert. Convert the "Original Config" to the "Target Config" based on user requirements. Only handle middleware related configurations (Docker Compose, Helm, K8s YAML, Nginx, Redis).

# Known Info
1. Original Config Type: {config_type};
2. Original Structured Config: {parsed_config};
3. User Core Requirement: {user_query}

# Conversion Rules
1. Target Type Extraction: Identify target type from the question;
2. Config Completeness: Output complete and usable target configuration;
3. Format Specs:
   - Wrap config in code block (```yaml or ```json);
   - Add "Brief Explanation within 3 points" after config.
4. Boundary Handling:
   - Prompt if target type not identified;
   - Prompt supported scenarios if conversion not supported;
   - Use reasonable default values for missing fields and annotate.

# Output Requirements
- First output the converted complete configuration;
- Then output brief explanation;
- No extra text.
"""
        try:
            converted_config = client.call_model(prompt)
            converted_config = converted_config.replace("\n\n\n", "\n\n")
            if "```yaml" in converted_config and "```" not in converted_config.split("```yaml")[-1]:
                converted_config += "\n```"
            if "```json" in converted_config and "```" not in converted_config.split("```json")[-1]:
                converted_config += "\n```"
            if not converted_config or converted_config.strip() == "":
                converted_config = "❌ Config conversion failed: Failed to generate valid target config, please try describing your conversion requirement again."
        except Exception:
            converted_config = "❌ Config conversion failed: Service busy, please try again later."
        return {"answer": converted_config}
    return fn
