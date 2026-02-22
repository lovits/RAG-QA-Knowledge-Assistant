from ..state_types import QASystemState
from ..registry.intents import get_node_for_intent, get_pipeline_for_intent
from ..agents.intent import make_intent_agent
from ..agents.analyzer import make_analyzer_agent
from ..agents.generator import make_generator_agent
from ..agents.reviewer import make_reviewer_agent
from ..agents.pure_qa import make_pure_qa_agent
from ..agents.retrieval import make_retrieval_agent
from ..agents.config_parse import make_config_parse_agent
from ..agents.config_convert import make_config_convert_agent
from ..agents.fault_troubleshoot import make_fault_troubleshoot_agent
from ..agents.supervisor import make_supervisor_agent
from ..agents.security import make_security_agent

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None

def build_graph(client):
    if StateGraph is None:
        raise ImportError("langgraph not installed")

    workflow = StateGraph(QASystemState)

    # 1. Add Nodes
    workflow.add_node("intent_agent", make_intent_agent(client))
    workflow.add_node("supervisor_agent", make_supervisor_agent(client))
    workflow.add_node("analyzer_agent", make_analyzer_agent(client))
    workflow.add_node("retrieval_agent", make_retrieval_agent(client))
    workflow.add_node("pure_qa_agent", make_pure_qa_agent(client))
    workflow.add_node("generator_agent", make_generator_agent(client))
    workflow.add_node("reviewer_agent", make_reviewer_agent(client))
    workflow.add_node("config_parse_agent", make_config_parse_agent(client))
    workflow.add_node("config_convert_agent", make_config_convert_agent(client))
    workflow.add_node("fault_troubleshooting_agent", make_fault_troubleshoot_agent(client))
    workflow.add_node("security_agent", make_security_agent(client))

    # 2. Add Edges
    workflow.set_entry_point("intent_agent")

    def intent_branch(state: QASystemState) -> str:
        intent = state.get("intent", "Pure QA")
        return get_node_for_intent(intent)

    def next_after_analyzer(state: QASystemState) -> str:
        intent = state.get("intent", "Pure QA")
        pipeline = get_pipeline_for_intent(intent)
        try:
            idx = pipeline.index("analyzer_agent")
            if idx + 1 < len(pipeline):
                return pipeline[idx + 1]
        except ValueError:
            pass
        return "pure_qa_agent"
    
    # intent_agent -> (dynamic)
    workflow.add_conditional_edges(
        "intent_agent",
        intent_branch
    )

    # Dynamic pipeline construction logic
    # Assume fixed pipelines for now, actually better to dynamically chain
    # But StateGraph requires explicit edges.
    # We can use "supervisor" as a router or simple linear chain per intent.
    # Here we simplify:
    
    # Supervisor -> Analyzer
    workflow.add_edge("supervisor_agent", "analyzer_agent")

    # Analyzer -> (dynamic)
    workflow.add_conditional_edges(
        "analyzer_agent",
        next_after_analyzer
    )

    # Retrieval -> PureQA
    workflow.add_edge("retrieval_agent", "pure_qa_agent")
    
    # PureQA -> Generator
    workflow.add_edge("pure_qa_agent", "generator_agent")

    # Config Parse -> Config Convert
    workflow.add_edge("config_parse_agent", "config_convert_agent")
    # Config Convert -> Generator
    workflow.add_edge("config_convert_agent", "generator_agent")

    # Fault Troubleshooting -> Generator
    workflow.add_edge("fault_troubleshooting_agent", "generator_agent")

    # Generator -> Reviewer
    workflow.add_edge("generator_agent", "reviewer_agent")

    # Security -> END
    workflow.add_edge("security_agent", END)

    # Reviewer -> (Security or END)
    def after_reviewer(state: QASystemState) -> str:
        intent = state.get("intent", "Pure QA")
        if intent == "Config Conversion":
            return "security_agent"
        return END

    workflow.add_conditional_edges(
        "reviewer_agent",
        after_reviewer,
        {
            "security_agent": "security_agent",
            END: END
        }
    )

    return workflow.compile()
