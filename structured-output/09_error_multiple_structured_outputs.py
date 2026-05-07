

from __future__ import annotations

from typing import Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.7,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="用户的名字")
    email: str = Field(description="用户的邮箱")
    phone: str = Field(description="用户的电话")


class EventDetails(BaseModel):
    """活动详情"""
    event_name: str = Field(description="活动名称")
    date: str = Field(description="活动日期")
    location: str = Field(description="活动地点")

union_schema = {
    "oneOf": [
        ContactInfo.model_json_schema(),
        EventDetails.model_json_schema()
    ]
}

# 添加明确的 system prompt，让模型只选择一个结构
system_prompt = (
    "你需要从用户输入中提取信息。"
    "根据输入内容，选择以下**其中一个**结构返回，不要同时返回多个：\n"
    "1. ContactInfo - 当用户提供了联系人信息（姓名、邮箱、电话）时\n"
    "2. EventDetails - 当用户提供了活动信息（活动名称、日期、地点）时\n"
    "只选择最匹配的一个结构，不要同时返回两个。"
)

agent = create_agent(
    model=model,
    tools=[],
    # response_format=ToolStrategy(union_schema),
    response_format=ToolStrategy(Union[ContactInfo, EventDetails]),
    system_prompt=system_prompt,
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "提取信息：张三 (zhangsan@qq.com) 将于2024年3月15日"
                        "在北京参加AI技术大会"
                    ),
                }
            ]
        }
    )

    print("=== structured_response ===")
    print(result["structured_response"])
    print()
    print("=== tool messages in history ===")
    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage":
            print(msg.content)
