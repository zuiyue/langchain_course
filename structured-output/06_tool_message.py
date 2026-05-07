from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.redis import RedisSaver
from typing_extensions import TypedDict

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL

# model = init_chat_model(
#     model="deepseek:deepseek-chat",
#     temperature=1.5,
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL
# )


model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)

product_review_schmea={
    "type": "object",
    "description":"product review",
    "properties":{
        "rating":{
            "type": ["number","null"],
            "description":"rating of product",
            "minimum":1,
            "maximum":100
        },
        "sentiment":{
            "type": "string",
            "enum":["positive","negative"],
            "description":"sentiment of product",
        },
        "keywords":{
            "type": "array",
            "items":{"type":"string"},
            "description":"keywords of product",
        }
    },
    "required":["rating","sentiment","keywords"]
}


with RedisSaver.from_conn_string("redis://localhost:6379/0") as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        response_format=ToolStrategy(schema=product_review_schmea,tool_message_content="ok************"),
    )

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


    for msg in result["messages"]:
        if type(msg).__name__=="ToolMessage":
            print(msg)





