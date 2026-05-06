from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_deepseek import  ChatDeepSeek
from openai import base_url
from pyexpat.errors import messages

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


@tool
def get_weather(city:str)->str:
    """根据城市名获取对应的天气情况"""
    print("get weather..........")
    return f"{city}的天气很好，22°！"


with_tools_model=model.bind_tools([get_weather])

messages =[
    {
        "role":"user",
        "content":"北京的天气如何？"
    }
]

ai_msg=with_tools_model.invoke(messages)


messages.append(ai_msg)

if not ai_msg.tool_calls:
    print("no my_tool call")

print(ai_msg)
for tool in ai_msg.tool_calls:
    print(tool.get("name"))
    print(tool.get("args"))
    if "get_weather" == tool.get("name"):
        tool_msg=get_weather.invoke(tool)
        print(tool_msg)
        messages.append(tool_msg)



final_result =with_tools_model.invoke(messages)

print(final_result)




