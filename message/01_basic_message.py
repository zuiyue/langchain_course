from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
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


# result = model.invoke("你好")
#
#
# messages=[
#     SystemMessage("你是一个有用文学专家"),
#     HumanMessage("分析一下红楼梦的经典所在")
# ]
#
# result = model.invoke(messages)
#
# print(result)


messages=[
    {"role":"system","content":"你是一个有用文学专家"},
    {"role":"user","content":"分析一下红楼梦的经典所在"}
]

result = model.invoke(messages)

print(result)



