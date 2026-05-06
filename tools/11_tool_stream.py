

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore


@tool
def get_weather(city:str,runtime:ToolRuntime)-> str:
    """查询天气，实时输出信息"""
    writer = runtime.stream_writer

    writer(f"start weather query {city}")
    writer(f"end weather query {city}")

    return  f"Weather in {city} is suny"




builder = StateGraph(MessagesState)
builder.add_node("tools",ToolNode([get_weather]))

builder.add_edge(START,"tools")
builder.add_edge("tools",END)

store= InMemoryStore()

graph=builder.compile(store=store)

ai_message=AIMessage(content="",
                     tool_calls=[
                         {
                             "id":"call_state_1",
                             "name":"get_weather",
                             "args":{
                                 "city":"北京"

                             },
                             "type":"tool_call",
                         }
                     ])



for mode,chunk in graph.stream({"messages":[HumanMessage(content="run tool"),ai_message]},stream_mode=["updates","custom"]):
    print(mode,chunk)
