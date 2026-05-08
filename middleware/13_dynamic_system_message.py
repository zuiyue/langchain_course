from __future__ import annotations

from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL

deepseek = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

glm = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)



@wrap_model_call
def dynamic_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if "deepseek" in str(request.messages) :
        print("[model selector] choose deepseek")
        selected_model = deepseek
    else:
        print("[model selector] choose glm")
        selected_model = glm
    return handler(request.override(model=selected_model,system_message=SystemMessage(content="你只能回答英语相关的问题，其他问题无法回答"),))


agent = create_agent(
    model=deepseek,
    tools=[],
    middleware=[dynamic_model],
)


if __name__ == "__main__":
    short_result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我把这句话翻译成英语：你今天忙吗？"}]}
    )
    print("[short conversation]", short_result["messages"][-1].content)


    print("*"*100)

    short_result = agent.invoke(
        {"messages": [{"role": "user", "content": "介绍一下glm"}]}
    )
    print("[short conversation]", short_result["messages"][-1].content)