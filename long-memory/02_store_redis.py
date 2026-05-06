from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.store.redis import RedisStore

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


@dataclass
class CustomContext:
    user_id:str


@tool
def get_user_profile(runtime:ToolRuntime[CustomContext])->str:
    """获取用户相关的信息"""
    namespace=("users",runtime.context.user_id)

    item=runtime.store.get(namespace,"profile")
    if item is None:
        return "No profile"

    profile=item.value

    return (
        f"name: {profile['name']}\n"
        f"email: {profile['email']}\n"
        f"skill: {profile['skill']}\n"
    )




with RedisStore.from_conn_string("redis://localhost:6379/0") as store:
    store.setup()

    # store.put(
    #     ("users","user_123"),
    #     "profile",
    #     {
    #         "name":"张三",
    #         "email":"777@qq.com",
    #         "skill":"langchain",
    #     }
    # )

    agent = create_agent(
        model=model,
        tools=[get_user_profile],
        store=store,
        system_prompt="你是一个使用工具帮用户完成任务的有用助手")



    result=agent.invoke({"messages":[{"role":"user","content":"我叫什么"}]},context=CustomContext(user_id="user_123"))


    print(result["messages"][-1].content)




