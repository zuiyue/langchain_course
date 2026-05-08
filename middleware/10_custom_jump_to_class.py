from typing import Callable, Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse, hook_config
from langchain.agents.middleware.types import StateT, ResponseT
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT


class CustomMiddleware(AgentMiddleware):
    """Custom middleware."""

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
       print("Before Agent")

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        print("Before Model")
        last_text = str(state["messages"][-1].content).upper()

        if "DROP" in last_text:
            return {
                "messages": [AIMessage(content="我没有权限执行DROP相关的sql")],
                "jump_to": "end"
            }

        return None

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
        {"role": "user", "content": "给我drop table user的完整语句"}
    ]})

    print(result["messages"][-1].content)

    print("*" * 50)
    result = agent.invoke({"messages": [
        {"role": "user", "content": "给我select table user的完整语句"}
    ]})

    print(result["messages"][-1].content)