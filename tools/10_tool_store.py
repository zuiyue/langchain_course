from typing import Any

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore


@tool
def svae_user_info(user_id:str,user_info:dict[str,Any],runtime:ToolRuntime)-> str:
    """存储用户信息"""
    runtime.store.put(("users",),user_id,user_info)

    return f"Saved user{user_id}"


@tool
def get_user_info(user_id: str,  runtime: ToolRuntime) -> str:
    """获取用户信息"""
    item=runtime.store.get(("users",), user_id,)

    return str(item.value) if item else "Unknown user"

builder = StateGraph(MessagesState)
builder.add_node("tools",ToolNode([svae_user_info,get_user_info]))

builder.add_edge(START,"tools")
builder.add_edge("tools",END)

store= InMemoryStore()

graph=builder.compile(store=store)

ai_message=AIMessage(content="",
                     tool_calls=[
                         {
                             "id":"call_state_1",
                             "name":"svae_user_info",
                             "args":{
                                 "user_id":"abc123",
                                 "user_info":{
                                     "name":"张三",
                                     "age":25,
                                     "email":"7878@qq.com"
                                 }
                             },
                             "type":"tool_call",
                         }
                     ])


result=graph.invoke({ "messages":[HumanMessage(content="run tool"),ai_message]})

print(result["messages"])

print(result["messages"][-1].content)
print("***************")


ai_message=AIMessage(content="",
                     tool_calls=[
                         {
                             "id":"call_state_2",
                             "name":"get_user_info",
                             "args":{
                                 "user_id":"abc123",
                             },
                             "type":"tool_call",
                         }
                     ])


result=graph.invoke({ "messages":[HumanMessage(content="run tool"),ai_message]})

print(result["messages"])

print(result["messages"][-1].content)