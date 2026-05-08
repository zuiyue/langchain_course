from typing import Any, Callable, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model, after_model, before_agent, wrap_model_call, \
    wrap_tool_call, ModelRequest, ModelResponse, after_agent
from langchain.agents.middleware.types import ResponseT, ExtendedModelResponse
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


class CustomState(AgentState):
    model_call_count:NotRequired[int]
    user_id:NotRequired[str]

@before_model(state_schema=CustomState)
def custom_count_model_call(state:AgentState,runtime:Runtime) -> dict[str, Any] | None:
    return {"model_call_count":state.get("model_call_count",0)+1}

@before_model(state_schema=CustomState)
def custom_before_model(state:AgentState,runtime:Runtime) -> dict[str, Any] | None:
    count = state.get("model_call_count")
    if count >1:
        print("调用模型次数超限")
    else:
        print("Before model")



@after_model(state_schema=CustomState)
def custom_after_model(state:AgentState,runtime:Runtime) -> dict[str, Any] | None:
    print("After Model")



def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"




agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[custom_count_model_call,custom_before_model,custom_after_model],)

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"北京天气如何？"}
    ],
        "model_call_count":0,
        "user_id":"user_123"
    })


    print(result.get("model_call_count"))
    print(result.get("user_id"))
    print(result["messages"][-1].content)