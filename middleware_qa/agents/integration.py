from ..state_types import QASystemState

def make_result_integration_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        answer = state.get("answer", "").strip()
        intent = state.get("intent", "Pure QA")
        if not answer:
            final_answer = "Sorry, failed to generate a valid answer. Please try adding more details or rephrasing."
            return {"answer": final_answer}
        if "❌" in answer and len(answer) < 50:
            return {"answer": answer}
        security = state.get("security_report", "").strip()
        prompt = f"""
# Task
Organize the raw answer of middleware QA into a clear, practical, and professional final response, differentiated by user intent. If security audit info exists, append security warnings.

# Known Info
1. User Intent: {intent};
2. Raw Answer: {answer}.
3. Security Audit: {security}

# Organization Rules
## 1. Format Standardization
- Combination of natural language + structured content;
- Use backticks for commands;
- Use code blocks for configuration;
- Use bullet points for steps.

## 2. Differentiated Processing by Intent
### Pure QA
- Put core answer first, add usage scenarios for commands;
- Add 1-2 best practice suggestions.
### Config Conversion
- Put configuration code block at the very beginning;
- Add 1-2 usage instructions;
- Annotate key parameters.
### Troubleshooting
- Break down solutions into steps, explaining purpose;
- Add 1 prevention suggestion.

## 3. Boundary Constraints
- Do not add core info not present in raw answer;
- Do not modify key content of raw answer;
- Total length under 300 words.

# Output Requirements
Directly output the final answer; if security audit info exists, add a "Security Audit" section at the end, briefly listing risks and suggestions.
"""
        try:
            final_answer = client.call_model(prompt)
            final_answer = final_answer.replace("\n\n\n", "\n\n")
            if "```yaml" in final_answer and "```" not in final_answer.split("```yaml")[-1]:
                final_answer += "\n```"
            if "```json" in final_answer and "```" not in final_answer.split("```json")[-1]:
                final_answer += "\n```"
        except Exception:
            final_answer = answer
        ir = state.get("intermediate_results", {})
        src = ir.get("retrieval_source", "")
        cnt = ir.get("retrieval_docs_count", 0)
        if src:
            final_answer = f"{final_answer}\n(Source: {src}; Hits: {cnt})"
        return {"answer": final_answer}
    return fn
