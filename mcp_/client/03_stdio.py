

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


from langchain_mcp_adapters.client import MultiServerMCPClient


from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

async def main():
    model = init_chat_model(
        model="glm-5.1",
        model_provider="openai",
        api_key=ZHIPU_API_KEY,
        base_url=ZHIPU_BASE_URL
    )

    mcp_config={
                "hello_mcp": {
                    "transport": "stdio",
                    "command":sys.executable,
                    "args":[str(Path(__file__).parent.parent / "server" / "03_stdio.py")],
                }
             }

    mcp_client=MultiServerMCPClient(mcp_config)

    tools= await  mcp_client.get_tools()


    for i, tool in enumerate(tools):
        print(f"{i}. {tool.name}")
        print(f"{i}. {tool.description}")
        print(f"{i}. {tool.args_schema}")


    agent = create_agent(
        model=model,
        tools=tools,
    )


    result = await agent.ainvoke({"messages":[{"role":"user","content":"请先张三打招呼"}]})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())