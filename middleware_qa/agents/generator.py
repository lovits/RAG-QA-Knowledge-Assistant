from ..state_types import QASystemState


def make_generator_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        intent = state.get("intent", "Pure QA")
        analysis = state.get("analysis", "")
        config_snippet = state.get("config_snippet", "")
        prompt = f"""
You are a generator. Generate a preliminary answer based on intent and analysis.

Input:
- Intent: {intent}
- Analysis: {analysis}
- Config Snippet: {config_snippet}

Generation Specs:
- Output ONLY the final answer, do not include prompt repetition or meta-info.
- Config Conversion: First provide the complete code block of the target configuration.
- Troubleshooting: Use numbered steps, explaining the purpose of each step.
- Use command lines and code blocks appropriately.
- Control length, suggested under 300 words.
"""
        answer = client.call_model(prompt)
        return {**state, "answer": answer}
    return fn
