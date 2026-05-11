

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.callbacks import Callbacks

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
                    "transport": "sse",
                    "url":"http://127.0.0.1:8080/sse"
                }
             }

    async def on_progress(*args, **kwargs):
        print(f"11111-->args={args}")
        print(f"22222-->kwargs={kwargs}")
    async def on_logging_message(*args, **kwargs):
        print(f"33333-->args={args}")
        print(f"44444-->kwargs={kwargs}")


    callbacks=Callbacks(
        on_logging_message=on_logging_message,
        on_progress=on_progress
    )
    mcp_client=MultiServerMCPClient(mcp_config,callbacks=callbacks)

    tools= await  mcp_client.get_tools()


    agent = create_agent(
        model=model,
        tools=tools,
    )


    result = await agent.ainvoke({"messages":[{"role":"user","content":"请帮我进行数据缝隙和报告生成"}]})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())