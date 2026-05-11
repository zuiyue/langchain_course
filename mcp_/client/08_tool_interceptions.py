

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

async def inject_user_context(request:MCPToolCallRequest,handler):

    context=request.runtime.context if request.runtime.context else {}
    user_id=context.get("user_id","匿名")

    print(f"inject_user_context：{user_id},调用工具：{request.name},server_name:{request.server_name}")

    return await handler(request)

async def log_exe_time(request:MCPToolCallRequest,handler):

    import time
    start_time = time.time()

    result =  await handler(request)

    all_time = time.time()-start_time
    print(f"工具{request.name}执行完成，耗时：{all_time}")

    return result

async def validate_params(request:MCPToolCallRequest,handler):

    print(f"参数校验：{request.args}")
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
                    tool_interceptors=[log_exe_time,inject_user_context,validate_params] )

    tools= await  mcp_client.get_tools()


    agent = create_agent(
        model=model,
        tools=tools,
    )


    result = await agent.ainvoke({"messages":[{"role":"user","content":"计算：8+9,在计算5*6"}]},context={"user_id":"user_123"})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())