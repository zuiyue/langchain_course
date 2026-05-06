from pprint import pprint
from tkinter.font import names

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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


sytem_msg=SystemMessage("你是一个有用的助手")
human_msg=HumanMessage(
    content="帮我写一首春天的诗",
    name="zhansan",
    id="user_001",
)

ai_msg=AIMessage(
    content="需要调用该天气查询的工具",
    id="ai_001"
)


too_msg=ToolMessage(
    content="这是一个查询天气的方法",
    tool_call_id="call_123",
    names="get_weather",
    artifact={
        "source":"langchain",
        "score":99
    }

)



messages = [sytem_msg, human_msg, ai_msg, too_msg]


for idx,msg in enumerate(messages):
    print(f"{idx}: {type(msg).__name__}")
    pprint(msg.model_dump())
    print("*"*50)

