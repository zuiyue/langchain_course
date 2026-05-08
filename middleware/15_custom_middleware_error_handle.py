from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model, after_model, before_agent, wrap_model_call, \
    wrap_tool_call, ModelRequest, ModelResponse, after_agent
from langchain.agents.middleware.types import ResponseT, ExtendedModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)

@tool
def divide(a:float, b:float) -> float:
    """完成除法运算"""
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b




@wrap_tool_call
def custom_wrap_tool_call(request: ToolCallRequest,
                   handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
    print("warp tool call before")

    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            tool_call_id=request.tool_call.get("id","fallback-tool-call"),
            content=f"error: {e}",
            status="error",
            name=request.tool_call["name"]
        )






agent = create_agent(
    model=model,
    tools=[divide],
    middleware=[custom_wrap_tool_call],)

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"计算 10/0"}
    ]})

    print(result["messages"][-1].content)