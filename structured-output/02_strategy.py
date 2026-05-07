from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.redis import RedisSaver
from pydantic import BaseModel, Field

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)



class ConactInfo(BaseModel):
    name: str=Field(description="用户的名字")
    email: str=Field(description="用户的邮箱")
    phone: str=Field(description="用户的电话")



with RedisSaver.from_conn_string("redis://localhost:6379/0") as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        # response_format=ProviderStrategy(ConactInfo),
        response_format=ToolStrategy(ConactInfo),
        system_prompt="你是一个使用工具帮用户完成任务的有用助手")

    config = {
        "configurable": {
            "thread_id": "thread_001"
        }
    }



    result=agent.invoke({"messages":[{"role":"user","content":"帮我提取信息，我叫张三，来自北京，你可以给个发信息到7878@qq.com，也可以给我打电话：（010）99998888"}]},config=config)


    structured=result["structured_response"]


    print(result["messages"][-1].content)
    print("*********")
    print(structured)




