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

gathered=None

for chunk in  model.stream("解读一下最近的低空经济政策"):
    if chunk.text:
        print(chunk.text)
    gathered=chunk if gathered is None else gathered + chunk


if gathered is not None:
    print(gathered)