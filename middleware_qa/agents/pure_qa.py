from ..state_types import QASystemState

def make_pure_qa_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        user_query = state["user_query"].strip()
        ir = state.get("intermediate_results", {})
        src = ir.get("retrieval_source", "")
        cnt = ir.get("retrieval_docs_count", 0)
        uq = user_query
        if cnt == 0:
            uq = f"{user_query}\n⚠️ No relevant background knowledge found. Please state clearly if unable to answer, do not fabricate information!"
        prompt = f"""
# Role
You are a senior middleware expert, proficient in theory, commands, configuration, and best practices for Docker, K8s, Helm, Nginx, Redis, MySQL, etc. You only answer questions related to middleware.

# Answer Requirements
1. Accuracy: Prioritize answers based on official documentation and industry standards. Do not fabricate commands/configurations. Mark uncertain content as "To be verified".
2. Conciseness: Avoid redundancy. Use bullet points (numbered/symbols). Put core information first.
3. Practicality:
   - If asked about commands: Provide "Command + Parameter Explanation + Usage Scenario". Format: `command` -> Explanation (Scenario).
   - If asked about theory/steps: Break down into steps, highlight key steps.
   - If asked about configuration: Provide core configuration snippets, annotate key parameters.
4. Boundary Handling: For non-middleware questions, reply directly: "Sorry, I focus only on middleware-related questions (Docker, K8s, Helm, Nginx, etc.). Please ask relevant content."

# User Question
{uq}
"""
        try:
            answer = client.call_model(prompt)
            answer = answer.replace("```", "").strip()
            if src == "fallback":
                answer = f"[Note: Answer based on local documents, information may be incomplete]\n{answer}"
            if not answer or answer.strip() == "":
                answer = "Sorry, failed to get a valid answer. Please try rephrasing your question."
        except Exception:
            answer = "Sorry, the service is currently busy. Please try again later."
        return {**state, "answer": answer}
    return fn
