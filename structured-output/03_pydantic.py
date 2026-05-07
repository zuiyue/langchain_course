from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.redis import RedisSaver
from typing_extensions import TypedDict

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)



# @dataclass
# class ProductReviewInfo:
#     rating:int|None
#     sentiment:Literal["positive","negative"]
#     key_point:list[str]




class ProductReviewInfo(TypedDict):
    rating:int|None
    sentiment:Literal["positive","negative"]
    key_point:list[str]



with RedisSaver.from_conn_string("redis://localhost:6379/0") as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        response_format=ToolStrategy(ProductReviewInfo))

    config = {
        "configurable": {
            "thread_id": "thread_001"
        }
    }



    result=agent.invoke({"messages":[{"role":"user",
                                      "content":("分析这个review: Great product,5/5."
                                                 "Fast shipping ,but expensive"
                                                 )
                                      }]},config=config)


    structured=result["structured_response"]


    print(result["messages"][-1].content)
    print("*********")
    print(structured)




