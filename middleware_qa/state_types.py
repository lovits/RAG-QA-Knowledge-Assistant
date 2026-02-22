from typing_extensions import TypedDict

class QASystemState(TypedDict):
    user_query: str
    intent: str
    config_snippet: str
    parsed_config: str
    answer: str
    intermediate_results: dict
    history: list
