from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_model = init_chat_model(
    "deepseek:deepseek-reasoner",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)



@tool(return_direct=True)
def get_weather(city:str)->str:
    """ 根据城市名称获取天气情况 """
    print("get_weather 被调用了！")
    return f"{city} 今天天气晴朗，22°***--++"

agent = create_agent(
    model=deepseek_model,
    tools=[get_weather],
    system_prompt="你是一个有用的助手",
)

result = agent.invoke({"messages":[{"role":"user","content":"今天北京天气怎么样？"}]})

print(result["messages"][-1].content)


