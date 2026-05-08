from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


class TraceMiddleware(AgentMiddleware):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"{self.label} -> before_agent")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"{self.label} -> before_model")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        print(f"{self.label} -> wrap_model_call (enter)")
        response = handler(request)
        print(f"{self.label} -> wrap_model_call (exit)")
        return response

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
        print(f"{self.label} -> wrap_tool_call (enter)")
        response = handler(request)
        print(f"{self.label} -> wrap_tool_call (exit)")
        return response

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"{self.label} -> after_model")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"{self.label} -> after_agent")
        return None

class A(TraceMiddleware):
    def __init__(self, label: str) -> None:
        super().__init__(label)
class B(TraceMiddleware):
    def __init__(self, label: str) -> None:
        super().__init__(label)

class C(TraceMiddleware):
    def __init__(self, label: str) -> None:
        super().__init__(label)


def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[A("m1"), B("m2"), C("m3")],
)


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "北京天气如何"}]}
    )
    print(result["messages"][-1].content)
