from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_deepseek import  ChatDeepSeek
from openai import base_url
from pyexpat.errors import messages

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


@tool
def get_weather(city:str)->str:
    """根据城市名获取对应的天气情况"""
    print("get weather..........")
    return f"{city}的天气很好，22°！"



@tool
def add(a:int ,b:int )->int:
    """加和两个整数"""
    print("add..........")
    return a+b

#
# with_tools_model=model.bind_tools([add,get_weather],tool_choice="get_weather")
#
# result=with_tools_model.invoke("hi,1+1")
#
# print(result)



# with_tools_model=model.bind_tools([get_weather],tool_choice="get_weather",parallel_tool_calls=False)
#
# result= with_tools_model.invoke("北京和上海的天气如何？")

#[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_00_dH8uc4uv2HhLHxwqEg0nCiol', 'type': 'tool_call'}, {'name': 'get_weather', 'args': {'city': '上海'}, 'id': 'call_01_cYAf6DuSaHIvjhqXA2T0AdcD', 'type': 'tool_call'}]
#[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_00_kvcB0PzF0hig2yhkkYfQLJob', 'type': 'tool_call'}, {'name': 'get_weather', 'args': {'city': '上海'}, 'id': 'call_01_ZsjLbxz8dByXNcoHdN2Gf8bm', 'type': 'tool_call'}]
# print(result.tool_calls)



with_tools_model=model.bind_tools([add,get_weather])


gathered=None
for chunk in with_tools_model.stream("北京天气如何"):
    gathered=chunk if gathered is None else gathered + chunk

if gathered is not None:
    print("gathered:",gathered)