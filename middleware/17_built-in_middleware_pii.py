from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model, PIIMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)



agent = create_agent(
    model=model,
    tools=[],
    middleware=[PIIMiddleware("email",strategy="redact",apply_to_input=True),
                PIIMiddleware("credit_card",strategy="mask",apply_to_input=True)],)

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"我的邮箱是77777@qq.com,我的卡号是：4111-1111-1111-1111.帮我进行信息安全处理，给出处理后可以展示的邮箱。保留首尾字符"}
    ]})

    print(result["messages"][-1].content)

