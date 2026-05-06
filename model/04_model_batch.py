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



prompts =[
    "什么RAG？"
    "什么是tool calling?"
    "简要说明langchian的原理"
]


result = model.batch(prompts,config={"max_concurrency":3})
#
#
# for i,r in enumerate(result):
#     print(f"{i}. {r.text}")
result = model.batch_as_completed(prompts,config={"max_concurrency":3})


for t in enumerate(result):
    print(f"{t}")