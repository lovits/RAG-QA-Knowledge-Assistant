from typing import Dict, List

class IntentDef:
    def __init__(self, node_name: str, description: str, pipeline: List[str]):
        self.node_name = node_name
        self.description = description
        self.pipeline = pipeline

_REGISTRY: Dict[str, IntentDef] = {
    "Pure QA": IntentDef(
        "supervisor_agent",
        "No specific configuration snippet involved, only answer theory/commands/steps",
        ["supervisor_agent", "analyzer_agent", "retrieval_agent", "pure_qa_agent", "generator_agent", "reviewer_agent"]
    ),
    "Config Conversion": IntentDef(
        "supervisor_agent",
        "User provided middleware configuration and requested conversion to another format",
        ["supervisor_agent", "analyzer_agent", "config_parse_agent", "config_convert_agent", "generator_agent", "reviewer_agent", "security_agent"]
    ),
    "Troubleshooting": IntentDef(
        "supervisor_agent",
        "User provided error logs or exception description and requested solution",
        ["supervisor_agent", "analyzer_agent", "fault_troubleshooting_agent", "generator_agent", "reviewer_agent"]
    ),
}

def allowed_intents() -> List[str]:
    return list(_REGISTRY.keys())

def get_node_for_intent(intent: str) -> str:
    if intent in _REGISTRY:
        return _REGISTRY[intent].node_name
    return "supervisor_agent"

def allowed_nodes() -> List[str]:
    return list(set(d.node_name for d in _REGISTRY.values()))

def get_intent_descriptions() -> str:
    lines = []
    for intent, definition in _REGISTRY.items():
        lines.append(f"- {intent}: {definition.description}")
    return ";\n   ".join(lines) + "."

def get_pipeline_for_intent(intent: str) -> List[str]:
    if intent in _REGISTRY:
        return _REGISTRY[intent].pipeline
    return ["supervisor_agent", "analyzer_agent", "generator_agent", "reviewer_agent"]

def all_pipelines() -> List[List[str]]:
    return [d.pipeline for d in _REGISTRY.values()]

def first_nodes() -> List[str]:
    return list({pipeline[0] for pipeline in all_pipelines() if pipeline})

def all_pipeline_nodes() -> List[str]:
    nodes = []
    for p in all_pipelines():
        nodes.extend(p)
    return list(set(nodes))

def all_edges() -> List[tuple]:
    edges = []
    for p in all_pipelines():
        for i in range(len(p) - 1):
            edges.append((p[i], p[i+1]))
    return edges

def last_nodes() -> List[str]:
    last = []
    for p in all_pipelines():
        if p:
            last.append(p[-1])
    return list(set(last))

def next_nodes_after(node_name: str) -> List[str]:
    nexts: List[str] = []
    for p in all_pipelines():
        for i in range(len(p) - 1):
            if p[i] == node_name:
                nexts.append(p[i + 1])
    return list(set(nexts))
