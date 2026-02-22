from ..state_types import QASystemState

def make_reviewer_agent(client):
    def fn(state: QASystemState) -> QASystemState:
        intent = state.get("intent", "Pure QA")
        answer = state.get("answer", "")
        ir = state.get("intermediate_results", {})
        src = ir.get("retrieval_source", "")
        cnt = ir.get("retrieval_docs_count", 0)
        stricter = (cnt == 0) or (src == "fallback")
        prompt = f"""
You are a reviewer. Review and polish the answer quality.

Input:
- Intent: {intent}
- Answer: {answer}

Output Specs:
- Output ONLY the polished final answer.
- Retain core content, improve structure and wording.
- Control length, max 300 words.
{"- When information is insufficient (no hits or fallback retrieval), strictly forbid fabrication, clearly state that more context is needed;" if stricter else "- Do not add disclaimers or extensions unrelated to the question;"}
"""
        refined = client.call_model(prompt)
        return {**state, "answer": refined}
    return fn
