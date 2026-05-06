from langchain.chat_models import init_chat_model
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


messages =[
    {"role":"system","content":"你是一个有用的助手"},
    {"role":"user","content":"我正在学习langchain"},
    {"role":"assistant","content":"你首先想要了解什么？"},
    {"role":"user","content":"什么时候tool calling ?"},

]
result = model.invoke(messages)
print(result)
print("*"*50)