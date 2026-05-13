
import re
from typing import Callable, Literal
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# ========== 1) 模型 ==========
model = init_chat_model(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.2,
)


# ========== 2) 状态 ==========
class MultiAgentState(AgentState):
    """在默认 AgentState 基础上新增 active_agent，用于记录当前专员。"""

    active_agent: NotRequired[str]


# ========== 3) 意图识别工具函数 ==========
RESET_PASSWORD_KEYWORDS = (
    "重置密码",
    "忘记密码",
    "登录失败",
    "无法登录",
    "登录不上",
    "密码重置",
)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _extract_email(text: str) -> str | None:
    match = EMAIL_PATTERN.search(text or "")
    return match.group(0) if match else None


def _latest_human_text(messages) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


# ========== 4) 中间件 ==========
@wrap_tool_call
def tool_call_logger(request, handler):
    tool_name = request.tool_call.get("name", "unknown")
    tool_args = request.tool_call.get("args", {})
    print(f"\n[中间件] 工具调用 -> {tool_name}, args={tool_args}")
    return handler(request)


@wrap_model_call
def force_reset_password_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    当满足以下条件时，给 support_agent 注入“必须先调 reset_password”的强约束：
    - 当前可用工具里包含 reset_password（即 support_agent 回合）；
    - 最近一条用户消息包含邮箱；
    - 最近一条用户消息包含重置/登录失败类关键词。
    """
    tool_names = {getattr(t, "name", None) for t in (request.tools or [])}
    if "reset_password" not in tool_names:
        return handler(request)

    latest_text = _latest_human_text(request.messages)
    email = _extract_email(latest_text)
    has_reset_intent = any(keyword in latest_text for keyword in RESET_PASSWORD_KEYWORDS)

    if not email or not has_reset_intent:
        return handler(request)

    force_rule = (
        f"执行规则：用户已明确要求重置密码，邮箱为 {email}。"
        "你必须先调用 reset_password(account_email=该邮箱)，"
        "本轮不要再次询问确认。"
    )

    base_system_text = request.system_message.text if request.system_message else ""
    merged_system = SystemMessage(content=f"{base_system_text}\n\n{force_rule}".strip())

    # 某些模型后端不支持 tool_choice='required'，做兼容降级。
    try:
        overridden = request.override(
            system_message=merged_system,
            tool_choice="required",
        )
        return handler(overridden)
    except Exception:
        fallback = request.override(system_message=merged_system)
        return handler(fallback)


# ========== 5) 业务工具（桩函数） ==========
@tool
def get_pricing(plan: Literal["basic", "pro", "enterprise"] = "pro") -> str:
    """查询套餐价格。"""

    price_table = {
        "basic": "99 元/月",
        "pro": "199 元/月",
        "enterprise": "499 元/月",
    }
    return f"{plan} 套餐当前价格：{price_table[plan]}"


@tool
def reset_password(account_email: str) -> str:
    """重置账户密码（桩函数）。"""
    print("---->重置账户密码（桩函数）")
    return f"已为 {account_email} 发起密码重置邮件，请在 10 分钟内查收。"


# ========== 6) Handoff 工具 ==========
@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    """将当前会话转接给销售专员。"""

    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )

    transfer_message = ToolMessage(
        content="已从技术支持转接到销售专员。",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_support(runtime: ToolRuntime) -> Command:
    """将当前会话转接给技术支持专员。"""

    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )

    transfer_message = ToolMessage(
        content="已从销售转接到技术支持专员。",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


# ========== 7) 创建两个子智能体 ==========
sales_agent = create_agent(
    model=model,
    tools=[get_pricing, transfer_to_support],
    system_prompt=(
        "你是销售专员，擅长回答套餐、价格、购买与升级问题。"
        "当用户询问技术故障（登录失败、报错、无法使用）时，"
        "必须调用 transfer_to_support 转接。"
    ),
    middleware=[tool_call_logger],
)

support_agent = create_agent(
    model=model,
    tools=[reset_password, transfer_to_sales],
    system_prompt=(
        "你是技术支持专员，擅长处理登录、报错、密码重置问题。"
        "当用户询问价格、套餐、购买时，必须调用 transfer_to_sales 转接。"
        "当用户已提供邮箱并明确要求重置密码时，"
        "必须直接调用 reset_password，不要先做二次确认。"
    ),
    middleware=[tool_call_logger, force_reset_password_call],
)


# ========== 8) 图节点与路由 ==========
def call_sales_agent(state: MultiAgentState):
    return sales_agent.invoke(state)


def call_support_agent(state: MultiAgentState):
    return support_agent.invoke(state)


def route_initial(state: MultiAgentState) -> Literal["sales_agent", "support_agent"]:
    """首次路由：根据 active_agent，默认进入 support_agent。"""

    return state.get("active_agent") or "support_agent"


def route_after_agent(
    state: MultiAgentState,
) -> Literal["sales_agent", "support_agent", "__end__"]:
    """如果已输出最终答复则结束，否则按 active_agent 继续。"""

    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            return "__end__"

    return state.get("active_agent") or "support_agent"


builder = StateGraph(MultiAgentState)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)

builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])

builder.add_conditional_edges(
    "sales_agent", route_after_agent, ["sales_agent", "support_agent", END]
)
builder.add_conditional_edges(
    "support_agent", route_after_agent, ["sales_agent", "support_agent", END]
)

graph = builder.compile()


# ========== 9) 演示运行 ==========
def print_trace(messages):
    print("\n--- 对话轨迹 ---")
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"[AI] 调用工具: {[call['name'] for call in msg.tool_calls]}")
            else:
                print(f"[AI] {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"[Tool] {msg.content}")
        else:
            msg_type = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            if msg_type == "human":
                print(f"[用户] {content}")
            else:
                print(f"[{msg_type}] {content}")


def run_case(title: str, user_text: str, active_agent: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"初始专员: {active_agent}")
    print(f"用户问题: {user_text}")

    final_state = graph.invoke(
        {
            "messages": [{"role": "user", "content": user_text}],
            "active_agent": active_agent,
        },
        config={"recursion_limit": 12},
    )

    print_trace(final_state["messages"])
    print(f"\n最终 active_agent: {final_state.get('active_agent')}")


if __name__ == "__main__":
    #场景 1：先走支持，再 handoff 到销售
    run_case(
        title="场景 1：技术支持专员收到价格问题 -> 转接销售",
        user_text="我想了解 pro 套餐多少钱，能给个报价吗？",
        active_agent="support_agent",
    )

    # 场景 2：先走销售，再 handoff 到支持
    run_case(
        title="场景 2：销售专员收到登录故障 -> 转接技术支持",
        user_text="我账户一直登录失败，帮我重置密码到 test@example.com",
        active_agent="sales_agent",
    )