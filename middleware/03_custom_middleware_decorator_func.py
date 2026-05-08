from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model, after_model, before_agent, wrap_model_call, \
    wrap_tool_call, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT, ExtendedModelResponse, after_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
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


def custom_before_model(state:AgentState,runtime:Runtime) -> dict[str, Any] | None:
    print("Before Model")

def custom_after_model(state:AgentState,runtime:Runtime) -> dict[str, Any] | None:
    print("After Model")

def custom_before_agent( state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
   print("Before Agent")

def custom_after_agent( state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("After Agent")


def custom_wrap_model_call( request: ModelRequest,
                    handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse | AIMessage | ExtendedModelResponse:


    print("warp model call before")
    result =handler(request)
    print("warp model call after")
    return result


def custom_wrap_tool_call(request: ToolCallRequest,
                   handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
    print("warp tool call before")
    result = handler(request)
    print("warp tool call after")
    return result




def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[
        before_model(custom_before_model),
        after_model(custom_after_model),
        before_agent(custom_before_agent),
        after_agent(custom_after_agent),
        wrap_model_call(custom_wrap_model_call),
        wrap_tool_call(custom_wrap_tool_call)
    ])

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"北京天气如何？"}
    ]})

    print(result["messages"][-1].content)