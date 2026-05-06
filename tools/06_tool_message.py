
from langchain.tools import tool,ToolRuntime
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


@tool
def get_use_message(runtime:ToolRuntime)-> str:
    """获取用户的message"""
    messages=runtime.state["messages"]
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return f" found user message: {message.content}"

    return "no message"


builder = StateGraph(MessagesState)
builder.add_node("tools",ToolNode([get_use_message]))


builder.add_edge(START,"tools")
builder.add_edge("tools",END)

graph=builder.compile()



ai_message=AIMessage(content="",
                     tool_calls=[
                         {
                             "id":"call_state_1",
                             "name":"get_use_message",
                             "args":{},
                             "type":"tool_call",

                         }
                     ])


result=graph.invoke({
    "messages":[HumanMessage(content="你好，我是张三"),ai_message]
})


print(result["messages"][-1].content)