from typing import Callable, Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse
from langchain.agents.middleware.types import StateT, ResponseT, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT


class CustomState(AgentState):
    model_call_count:NotRequired[int]
    user_id:NotRequired[str]

class CustomMiddleware(AgentMiddleware):
    """Custom middleware."""
    state_schema=CustomState

    def before_agent(self, state: CustomState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        count = state.get("model_call_count")
        if count > 1:
            print("调用模型次数超限")
        else:
            print("Before model")

    def before_model(self, state: CustomState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        print("Before Model")
        return {"model_call_count": state.get("model_call_count", 0) + 1}

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        print("After Model")

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        print("After Agent")



    def wrap_model_call(self, request: ModelRequest[ContextT],
                        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:


        print("warp model call before")
        result =handler(request)
        print("warp model call after")
        return result


    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
        print("warp tool call before")
        result = handler(request)
        print("warp tool call after")
        return result



from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)


def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[CustomMiddleware()])

if __name__ == "__main__":
    result = agent.invoke({"messages": [
        {"role": "user", "content": "北京天气如何？"}
    ],
        "model_call_count": 0,
        "user_id": "user_123"
    })

    print(result.get("model_call_count"))
    print(result.get("user_id"))
    print(result["messages"][-1].content)