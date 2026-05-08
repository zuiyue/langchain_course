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
    last_model_call_tokens:NotRequired[int]
    user_id:NotRequired[str]


@wrap_model_call(state_schema=CustomState)
def track_usage(request:ModelRequest,handle:Callable[[ModelRequest],ModelResponse]) -> ExtendedModelResponse:
    result=handle(request)
    tokens=sum( len(str(msg.content)) for msg in request.messages)

    return ExtendedModelResponse(
        model_response=result,
        command=Command(
            update={
                "last_model_call_tokens": tokens
            }
        )
    )


def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"




agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[track_usage],)

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"写一首春天的诗句"}
    ],
        "model_call_count":0,
        "user_id":"user_123"
    })


    print(result.get("last_model_call_tokens"))
    print(result.get("user_id"))
    print(result["messages"][-1].content)