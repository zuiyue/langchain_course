from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)


@dataclass
class UserContext:
    user_type:str


@dynamic_prompt
def prompt_by_user_type(request: ModelRequest[UserContext]) -> str:
    user_type=request.runtime.context.user_type
    print(f"user_type: {user_type}")
    if user_type =="VIP":
        return "你是一个专家助手，可以解答复杂的数学问题"
    return "你只是一个普通的助手，只能帮助用户解决天气问题，其他的无法回答"


def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"


agent = create_agent(
    model=model,
    tools=[get_weather],
    context_schema=UserContext,
    middleware=[prompt_by_user_type])

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"66的3倍是多少"}
    ]},
    context=UserContext(user_type="VIP")
    )

    print(result["messages"][-1].content)

    print("*"*50)
    result = agent.invoke({"messages": [
        {"role": "user", "content": "66的3倍是多少"}
    ]},
        context=UserContext(user_type="COMM")
    )

    print(result["messages"][-1].content)

    print("*" * 50)
    result = agent.invoke({"messages": [
        {"role": "user", "content": "北京天气如何"}
    ]},
        context=UserContext(user_type="COMM")
    )

    print(result["messages"][-1].content)