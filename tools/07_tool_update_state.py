
from langchain.tools import tool,ToolRuntime
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import tool_call
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


class UserState(MessagesState):
    user_name:str

@tool
def set_user_name(new_name:str,runtime:ToolRuntime)-> str:
    """更新状态中的用户名"""
    return Command(
        update={
            "user_name":new_name,
            "messages":[
                ToolMessage(content=f"设置用户名：{new_name}",tool_call_id=runtime.tool_call_id),
            ]
        }
    )

builder = StateGraph(UserState)
builder.add_node("tools",ToolNode([set_user_name]))


builder.add_edge(START,"tools")
builder.add_edge("tools",END)

graph=builder.compile()



ai_message=AIMessage(content="",
                     tool_calls=[
                         {
                             "id":"call_state_1",
                             "name":"set_user_name",
                             "args":{"new_name":"王武"},
                             "type":"tool_call",

                         }
                     ])


result=graph.invoke({
    "messages":[HumanMessage(content="请存储我的名称：王武"),ai_message],
    "user_name":"",

})


print(result["user_name"])
print(result["messages"])