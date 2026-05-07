

from __future__ import annotations

from typing import Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)


def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    return  f"{city} 天气晴朗，22°"


agent = create_agent(
    model=model,
    tools=[get_weather],

)



if __name__ == "__main__":
    user_input={
        "messages":[
            {"role":"user","content":"北京的天气如何？"}
        ]
    }

    for chunk in agent.stream(user_input,stream_mode="updates",version="v2"):
        for step,data in chunk["data"].items():
            print(f"{step}: {data}")