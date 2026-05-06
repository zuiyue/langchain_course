from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# @tool
# def get_user_profile(user_name:str)->str:
#     """根据用户名获取用户信息"""
#
#     profiles={
#         "张三":"张三精通python ai运用开发",
#         "李四":"喜欢使用langchain进行研发"
#     }
#
#     return profiles.get(user_name,f"没有该用户信息")


checkpointer=InMemorySaver()

agent = create_agent(
    model=model,
    # tools=[get_user_profile],
    checkpointer=checkpointer,
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

#result=agent.invoke({"messages":[{"role":"user","content":"你好，我叫张三"}]})
result=agent.invoke({"messages":[{"role":"user","content":"我叫什么"}]},config=config)




print(result["messages"][-1].content)




