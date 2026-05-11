

from __future__ import annotations

import asyncio


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
                    "transport": "sse",
                    "url":"http://127.0.0.1:8080/sse"
                }
             }





    mcp_client=MultiServerMCPClient(mcp_config)

    tools = await  mcp_client.get_tools()

    for i, tool in enumerate(tools):
        print(f"{i}. {tool.name}")
        print(f"{i}. {tool.description}")
        print(f"{i}. {tool.args_schema}")

    uri = ["file:///111111", "file:///222222"]
    resources = await  mcp_client.get_resources("hello_mcp", uris=uri)

    for resource in resources:
        print(resource.mimetype)
        print(resource.as_string())

    prompts= await  mcp_client.get_prompt("hello_mcp","code_review",arguments={"language":"python"})

    for i,prompt in enumerate(prompts):
        print(prompt.type)
        print(prompt.content)

if __name__ == "__main__":
    asyncio.run(main())