

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL


async def modify_requset_param(request:MCPToolCallRequest,handler):

    print(f"修改前参数：{request.args}")


    modify_param={
        "a":10,"b":20
    }

    modify_request=request.override(
        args=modify_param
    )
    print(f"修改后参数：{modify_request.args}")

    return  await handler(modify_request)





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
                    tool_interceptors=[modify_requset_param] )

    tools= await  mcp_client.get_tools()


    agent = create_agent(
        model=model,
        tools=tools,
    )


    result = await agent.ainvoke({"messages":[{"role":"user","content":"计算：8+9,在计算5*6"}]},context={"user_id":"user_123"})
    print(result["messages"])

if __name__ == "__main__":
    asyncio.run(main())