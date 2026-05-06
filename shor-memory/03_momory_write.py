from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


class CustomState(AgentState):
    user_name:str

class CustomContext(BaseModel):
    user_id:str


@tool
def update_user_info(runtime:ToolRuntime[CustomState,CustomContext])->Command:
    """查询用户信息病持久化到状态中"""

    user_id=runtime.context.user_id
    print("update_user_info",user_id)
    name="张三" if user_id=="user_100" else "Unknown user"
    return Command(
        update={
            "user_name":name,
            "messages":[
                ToolMessage(
                    content="成功获取用户信息",
                    tool_call_id=runtime.tool_call_id
                )
            ]  })

@tool
def greet(runtime:ToolRuntime[CustomState,CustomContext])->str|Command:
    """查询用户信息病持久化到状态中"""


    user_name=runtime.state.get("user_name")
    print("greet...",user_name)
    if user_name is None:
        return Command(
            update={
                "messages":[
                    ToolMessage(content="请先调用update_user_info",tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    return f"hello {user_name}"

checkpointer=InMemorySaver()

agent = create_agent(
    model=model,
    tools=[update_user_info,greet],
    checkpointer=checkpointer,
    state_schema=CustomState,
    context_schema=CustomContext,
    system_prompt="你是一个使用工具帮用户完成任务的有用助手")

config={
    "configurable":{
        "thread_id":"thread_001"
    }
}


config2={
    "configurable":{
        "thread_id":"thread_002"
    }
}


result=agent.invoke({"messages":[{"role":"user","content":"请先获取我的信息，在向我问好"}],

                     },config=config,context=CustomContext(user_id="user_100"))

for message in result['messages']:
    if hasattr(message,"content"):
        print(message.type,message.content)

print("*"*50)
result=agent.invoke({"messages":[{"role":"user","content":"我叫什么"}],

                     },config=config)


for message in result['messages']:
    if hasattr(message,"content"):
        print(message.type,message.content)



