from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict
from typing import Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


class ProductReview(BaseModel):
    """Analysis of a product review."""

    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment")
    key_points: list[str] = Field(description="Key points")


class ProductReviewInfo(TypedDict):
    rating:int|None
    sentiment:Literal["positive","negative"]
    key_point:list[str]



@dataclass
class ProductReviewInfo1:
    rating:int|None
    sentiment:Literal["positive","negative"]
    key_point:list[str]

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

# class CustomerComplaint(BaseModel):
#     """A customer complaint."""
#
#     issue_type: Literal["product", "service", "shipping", "billing"]
#     severity: Literal["low", "medium", "high"]
#     description: str

class ConactInfo(BaseModel):
    name: str=Field(description="用户的名字")
    email: str=Field(description="用户的邮箱")
    phone: str=Field(description="用户的电话")


agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(Union[ProductReviewInfo,ProductReviewInfo1,ConactInfo]),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": ("Analyze this review: Great product, 5/5. Fast shipping, but expensive."),
                }
            ]
        }
    )
    structured = result["structured_response"]
    print(type(structured))
    print(structured)

    result = agent.invoke({"messages": [{"role": "user",
                                         "content": "帮我提取信息，我叫张三，来自北京，你可以给个发信息到7878@qq.com，也可以给我打电话：（010）99998888"}]})

    structured = result["structured_response"]

    print(result["messages"][-1].content)
    print("*********")
    print(structured)
