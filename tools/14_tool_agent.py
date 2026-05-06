from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


@tool
def get_weather(city:str)-> str:
    """根据城市名查询天气"""
    print("get_weather....")
    return  f"Weather in {city} is suny"


@tool
def multiply(a:int,b:int)-> int:
    """两个整数相乘"""
    print("multiply....")
    return  a*b


if __name__ == "__main__":
    model = init_chat_model(
        model="deepseek:deepseek-reasoner",
        temperature=1.5,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )


    agent = create_agent(
        model=model,
        tools=[get_weather,multiply],
        system_prompt="你是一个使用工具帮用户完成任务的有用助手")

    result= agent.invoke({
        "messages":[{
            "role":"user",
            "content":"北京的天气如何？帮我计算5*6"
        }]
    })

    print(result)
    print("********************")
    print(result["messages"][-1].content)