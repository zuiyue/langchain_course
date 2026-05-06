from langchain.chat_models import init_chat_model
from langchain.tools import tool,ToolRuntime
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.tool import tool_call
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


class UserState(MessagesState):
    user_name:str


from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

@tool
def set_user_name(new_name:str,runtime:ToolRuntime)-> str:
    """更新状态中的用户名"""

    print("set_user_name.............")
    return Command(
        update={
            "user_name":new_name,
            "messages":[
                ToolMessage(content=f"设置用户名：{new_name}",tool_call_id=runtime.tool_call_id),
            ]
        }
    )

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


mode_with_tool=model.bind_tools([set_user_name])

def llm_node(state:UserState):
    result = mode_with_tool.invoke([SystemMessage(content="你是一个利用工具帮助用户完整任务的助手")]+state["messages"])
    return {"messages":[result]}



builder = StateGraph(UserState)
builder.add_node("llm_node",llm_node)
builder.add_node("tools",ToolNode([set_user_name]))


builder.add_edge(START,"llm_node")
builder.add_edge("llm_node","tools")
builder.add_edge("tools",END)

graph=builder.compile()



# ai_message=AIMessage(content="",
#                      tool_calls=[
#                          {
#                              "id":"call_state_1",
#                              "name":"set_user_name",
#                              "args":{"new_name":"王武"},
#                              "type":"tool_call",
#
#                          }
#                      ])


result=graph.invoke({
    "messages":[HumanMessage(content="请存储我的名称：王武")],
    "user_name":"",

})


print(result["user_name"])
print(result["messages"])