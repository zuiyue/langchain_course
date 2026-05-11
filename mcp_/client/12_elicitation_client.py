

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import ElicitResult

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
                    "url":"http://127.0.0.1:8001/sse"
                }
             }

    async def on_elicitation(mcp_context,params,context)->ElicitResult:


        message=params.message if hasattr(params, "message") else str(params)
        print(f"message={message}")


        while True:
            user_response=input("是否继续?(Y/N)").strip().upper()
            if user_response in ("Y", "N"):
                break
            print("请输入Y 或 N")

        if user_response == "Y":
            return ElicitResult(
                action="accept",
                content={"value":"是，继续"}
            )

        else:
            return ElicitResult(
                action="cancel",
                content={"value": "否，取消"}
            )



    callbacks=Callbacks(
        on_elicitation=on_elicitation
    )
    mcp_client=MultiServerMCPClient(mcp_config,callbacks=callbacks)

    tools= await  mcp_client.get_tools()


    agent = create_agent(
        model=model,
        tools=tools,
    )


    result = await agent.ainvoke({"messages":[{"role":"user","content":"请调用confirm_operation工具，执行任务：删除临时文件"}]})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())