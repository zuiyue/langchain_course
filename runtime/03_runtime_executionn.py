from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)


@dataclass
class Context:
    user_name:str
    user_id:str
    language:str


@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """获取当前用户信息"""
    user_name=runtime.context.user_name
    user_id=runtime.context.user_id
    language=runtime.context.language

    if language=="中文":
        return f"用户名：{user_name}，用户ID:{user_id},我喜欢中文"

    return f"用户名：{user_name}，用户ID:{user_id},我喜欢英文"




@before_model
def custom_before_model(state:AgentState,runtime:Runtime) -> dict[str, Any] | None:
    print("Before Model")
    user_name = runtime.context.user_name
    user_id = runtime.context.user_id
    language = runtime.context.language

    print(f"user_name:{user_name},user_id:{user_id},language:{language}")


    execution_info = runtime.execution_info

    thread_id = execution_info.thread_id
    run_id = execution_info.run_id
    node_attempt = execution_info.node_attempt

    print(f"thread_id:{thread_id},run_id:{run_id},node_attempt:{node_attempt}")





agent = create_agent(
    model=model,
    tools=[get_user_info],
    context_schema=Context,
    middleware=[custom_before_model]
)


if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "thread_001",
        },
        "run_id": "run001",
    }

    context = Context(user_name="张三",user_id="user_123",language="中文")

    result=agent.invoke({"messages":[
        {"role":"user","content":"获取我的用户信息"}
    ]},
        context=context, config=config
    )

    print(result["messages"][-1].content)

