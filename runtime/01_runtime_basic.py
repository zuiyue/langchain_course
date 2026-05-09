from dataclasses import dataclass
from typing import Callable, Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse
from langchain.agents.middleware.types import StateT, ResponseT
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT





from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)


@dataclass
class Context:
    user_name:str
    user_id:str


def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"


agent = create_agent(
    model=model,
    tools=[get_weather],
    context_schema=Context,
)


if __name__ == "__main__":

    context = Context(user_name="张三",user_id="user_123")

    result=agent.invoke({"messages":[
        {"role":"user","content":"北京天气如何？"}
    ]},
    context=context
    )

    print(result["messages"][-1].content)