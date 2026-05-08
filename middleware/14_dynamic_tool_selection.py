from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def weather_lookup(city: str) -> str:
    """Return a mocked weather sentence."""
    return f"{city}: sunny, 26C, light breeze."


@tool
def translate_to_chinese(text: str) -> str:
    """Mock translation."""
    return f"[中文翻译] {text}"


def _tool_name(tool_obj: Any) -> str:
    if isinstance(tool_obj, dict):
        return str(tool_obj.get("name", ""))
    return str(getattr(tool_obj, "name", ""))


@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    latest_text = str(request.messages[-1].content).lower()
    if "天气" in latest_text or "weather" in latest_text:
        allowed = {"weather_lookup"}
    elif "加" in latest_text or "sum" in latest_text or "计算" in latest_text:
        allowed = {"add_numbers"}
    elif "翻译" in latest_text or "translate" in latest_text:
        allowed = {"translate_to_chinese"}
    else:
        allowed = {"add_numbers", "weather_lookup", "translate_to_chinese"}

    selected = [t for t in request.tools if _tool_name(t) in allowed]
    if not selected:
        selected = request.tools

    print("[tool selector] tools:", [_tool_name(t) for t in selected])
    return handler(request.override(tools=selected))


agent = create_agent(
    model=model,
    tools=[add_numbers, weather_lookup, translate_to_chinese],
    middleware=[select_tools],
)


if __name__ == "__main__":
    math_result = agent.invoke(
        {"messages": [{"role": "user", "content": "请调用工具计算 23 + 19。"}]}
    )
    print("[math]", math_result["messages"][-1].content)

    weather_result = agent.invoke(
        {"messages": [{"role": "user", "content": "请调用工具查询北京天气。"}]}
    )
    print("[weather]", weather_result["messages"][-1].content)
