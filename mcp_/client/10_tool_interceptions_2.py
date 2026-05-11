

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langgraph.types import Command

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL


async def eary_termination(request:MCPToolCallRequest,handler):

    print(f"修改前参数：{request.args}")

    if "北京" in str(request.args):
        return Command(goto="__end__")

    return  await handler(request)





async def main():
    model = init_chat_model(
        model="glm-5.1",
        model_provider="openai",
        api_key=ZHIPU_API_KEY,
        base_url=ZHIPU_BASE_URL
    )

    mcp_config={
                "hello_mcp": {
                    "transport": "sse",
                    "url":"http://127.0.0.1:8001/sse"
                }
             }

    mcp_client=MultiServerMCPClient(mcp_config,
                    tool_interceptors=[eary_termination] )

    tools= await  mcp_client.get_tools()


    agent = create_agent(
        model=model,
        tools=tools,
    )


    result = await agent.ainvoke({"messages":[{"role":"user","content":"北京天气如何"}]},context={"user_id":"user_123"})
    print(result["messages"])

if __name__ == "__main__":
    asyncio.run(main())