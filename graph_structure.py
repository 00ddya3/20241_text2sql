from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import graph_function


class GraphState(TypedDict):
    user_question: str
    user_intent: str
    events_output: str
    top_k: str
    final_output: List[str]
    

def my_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("LLM_Router", graph_function.LLM_Router)
    workflow.add_node("LLM_event_list", graph_function.LLM_event_list)
    workflow.add_node("Retrieve", graph_function.Retrieve) 
    workflow.add_node("LLM_Final_Generate", graph_function.LLM_Final_Generate)  

    # Build graph
    workflow.add_edge(START, "LLM_Router")
    workflow.add_conditional_edges("LLM_Router",
                                   #lambda state: 
                                   lambda state: ("LLM_event_list" if state["user_intent"] == 'EVENT' 
                                                else "Retrieve" if state["user_intent"] == 'ANALYSIS'
                                                else "LLM_Final_Generate"),
                                   {"LLM_event_list": "LLM_event_list", "Retrieve": "Retrieve", "LLM_Final_Generate": "LLM_Final_Generate"}) #없으면 그래프 시각화가 엉뚱하게 됨..
    workflow.add_edge("LLM_event_list", END)
    workflow.add_edge("Retrieve", "LLM_Final_Generate")
    workflow.add_edge("LLM_Final_Generate", END)

    # Compile
    app = workflow.compile()
    return app


def my_graph_image(app):
    return app.get_graph().draw_mermaid_png()