from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)


@tool
def get_weather(city: str) -> str:
    """根据城市获取天气信息"""
    print("get weather.......", city)
    return f"{city}天气晴朗，22°"

@tool
def get_time(city: str) -> str:
    """根据城市获取当前时间"""
    print("get time.......", city)
    return f"{city}当前时间：14:55"


@before_model(tools=[get_weather, get_time])
def custom_before_model(state:AgentState,runtime:Runtime) -> dict[str, Any] | None:
    print("Before Model")



agent = create_agent(
    model=model,
    tools=[],
    middleware=[custom_before_model],)

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"北京天气如何？"}
    ]})

    print(result["messages"][-1].content)
    print("*"*100)

    result = agent.invoke({"messages": [
        {"role": "user", "content": "查询北京时间？"}
    ]})

    print(result["messages"][-1].content)
    print("*" * 100)

    result = agent.invoke({"messages": [
        {"role": "user", "content": "北京天气如何？再查询一下北京时间？"}
    ]})

    print(result["messages"][-1].content)
    print("*" * 100)