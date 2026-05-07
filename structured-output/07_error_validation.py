from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy, StructuredOutputValidationError, \
    MultipleStructuredOutputsError
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.redis import RedisSaver
from pydantic import BaseModel, Field
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


class ConactInfo(BaseModel):
    name: str=Field(description="用户的名字")
    age: int|None=Field(description="用户的年龄",ge=0,le=150)



def error_handle(error:Exception) -> str:
    if isinstance(error,StructuredOutputValidationError):
        return "StructuredOutputValidationError...."
    if isinstance(error,MultipleStructuredOutputsError):
        return "MultipleStructuredOutputsError...."
    return f"error:999999999999"

with RedisSaver.from_conn_string("redis://localhost:6379/0") as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        response_format=ToolStrategy(schema=ConactInfo,
                                     tool_message_content="ok************",
                                     # handle_errors="年龄必须在0-150之间"
                                     handle_errors=error_handle
    ),
    )

    config = {
        "configurable": {
            "thread_id": "thread_001"
        }
    }



    result=agent.invoke({"messages":[{"role":"user",
                                      "content":"帮我解析如下信息：我叫张三，今年160。不能自己指定任何数据"
                                      }]},config=config)


    structured=result["structured_response"]

    print(structured)
    print(result["messages"][-1].content)
    print("*********")







