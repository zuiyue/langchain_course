

from __future__ import annotations

import asyncio
from typing import Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk
from langgraph.config import get_stream_writer
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
    writer=get_stream_writer()
    writer({"stage":"start","city":city})
    writer({"stage":"fetched","city":city})

    return  f"{city} 天气晴朗，22°"


agent = create_agent(
    model=model,
    tools=[get_weather],

)

async  def main()->None:
    user_input={
        "messages":[
            {"role":"user","content":"北京的天气如何？"}
        ]
    }

    async  for chunk in agent.astream(user_input,stream_mode=["custom","messages"],version="v2"):
        if chunk["type"] != "messages":
            continue

        token,metadata=chunk["data"]
        if isinstance(token,AIMessageChunk):
            text=token.text
            if text:
                print(text,end="",flush=True)

            if token.tool_call_chunks:
                print(f"[tool_call_chunks] {token.tool_call_chunks}]")
                print(f"[source] {metadata.get('langgraph_node')}")


if __name__ == "__main__":
    asyncio.run(main())