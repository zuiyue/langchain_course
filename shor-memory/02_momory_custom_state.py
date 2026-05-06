from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


class CustomAgentState(AgentState):
    user_id:str
    preference:dict[str, Any]


@tool
def read_state_fields(runtime:ToolRuntime)->str:
    """从状态中读取自定义状态信息"""

    user_id=runtime.state.get("user_id","missing")
    preference=runtime.state.get("preference",{})

    print("read_state_fields.................")
    return f"user_id={user_id},preference={preference}"



checkpointer=InMemorySaver()

agent = create_agent(
    model=model,
    tools=[read_state_fields],
    checkpointer=checkpointer,
    state_schema=CustomAgentState,
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


result=agent.invoke({"messages":[{"role":"user","content":"调用工具read_state_fields，然后告诉我状态里有什么信息？"}],
                     "user_id":"user_100",
                     "preference":{"language":"zh-CN","address":"北京"},

                     },config=config)
print(result["messages"][-1].content)
print("*"*50)

result=agent.invoke({"messages":[{"role":"user","content":"再读一次当前状态"}]},config=config)

print(result["messages"][-1].content)




