from dataclasses import dataclass

from langchain.tools import tool,ToolRuntime
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import tool_call
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


@dataclass
class UserContext:
    user_id:str;


USER_DATA={
    "user_001":{"name":"张三","type":"premium","balance":5000},
    "user_002":{"name":"李四","type":"stand","balance":1200}
}

@tool
def get_user_info(runtime:ToolRuntime[UserContext])-> str:
    """更新状态中的用户名"""
    user_id=runtime.context.user_id
    user=USER_DATA.get(user_id)
    if user is None:
        return "user not fund"
    return (f"name:{user['name']}\nbalance:{user['balance']}\nType:{user['type']}\n")



builder = StateGraph(MessagesState,context_schema=UserContext)
builder.add_node("tools",ToolNode([get_user_info]))

builder.add_edge(START,"tools")
builder.add_edge("tools",END)

graph=builder.compile()

ai_message=AIMessage(content="",
                     tool_calls=[
                         {
                             "id":"call_state_1",
                             "name":"get_user_info",
                             "args":{"new_name":"王武"},
                             "type":"tool_call",
                         }
                     ])


result=graph.invoke({ "messages":[HumanMessage(content="查看我的账户信息"),ai_message]},context=UserContext(user_id="user_001"))

print(result["messages"])
print("***************")
print(result["messages"][-1].content)