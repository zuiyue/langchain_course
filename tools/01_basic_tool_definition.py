from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL



@tool
def search_database(query:str,limit:int=10)->str:
    """根据关键词查询数据库数据，返回指定的条数"""
    print("search_database.............")
    return f"found {limit} 条关于{query} 的数据"

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

model_with_tool=model.bind_tools([search_database])


messages=[
    HumanMessage("帮过我从数据库中查询3条关于langchain的数据")
]


result=model_with_tool.invoke(messages)


print(result)

messages.append(result)

for tool_call in result.tool_calls:
    tool_mag=search_database.invoke(tool_call)
    messages.append(tool_mag)

print("********************")
result=model_with_tool.invoke(messages)
print(result)
