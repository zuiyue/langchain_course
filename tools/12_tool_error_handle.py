

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore


@tool
def paese_number(number:int)-> str:
    """返回正数"""
    if number <= 0:
        raise ValueError("number must be > 0")
    return str(number)



def run(handle_tool_errors,label:str)->None:

    builder = StateGraph(MessagesState)
    builder.add_node("tools",ToolNode([paese_number],handle_tool_errors=handle_tool_errors))

    builder.add_edge(START,"tools")
    builder.add_edge("tools",END)


    graph=builder.compile()

    ai_message=AIMessage(content="",
                         tool_calls=[
                             {
                                 "id":"call_state_1",
                                 "name":"paese_number",
                                 "args":{
                                     "number":-1

                                 },
                                 "type":"tool_call",
                             }
                         ])


    try:
        print(f"-------{label}-------------")
        result=graph.invoke({"messages":[HumanMessage(content="run tool"),ai_message]})
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"tool error: {type(e).__name__}:{e}")





if __name__ == "__main__":
    run(False, "handle_tool_errors=False")
    print("*"*50)
    run(True, "handle_tool_errors=True")

    print("*" * 50)
    run("please privode a positive number ", "handle_tool_errors='custom string'")

    print("*" * 50)
    run(lambda e:f"custom handle:{e}", "handle_tool_errors=callbale")