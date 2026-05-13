

from typing import Literal



from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import START, MessagesState, StateGraph
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.7,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# ============================================
# 步骤 1：定义状态
# ============================================

class AgentState(MessagesState):
    """扩展消息状态，添加当前步骤/角色标识"""
    current_step: Literal["triage", "refunds", "tech_support"] = "triage"


# ============================================
# 步骤 2：定义交接工具
# ============================================

@tool
def transfer_to_refunds() -> str:
    """转接给退款专员。当用户要求退款时使用此工具。"""
    print("\n   交接：分类员 → 退款专员")
    return "已转接给退款专员"


@tool
def transfer_to_tech_support() -> str:
    """转接给技术支持专员。当用户有技术问题时使用此工具。"""
    print("\n   交接：分类员 → 技术支持专员")
    return "已转接给技术支持专员"


@tool
def transfer_back_to_triage() -> str:
    """转接回分类员。当需要切换话题时使用此工具。"""
    print("\n   交接：专员 → 分类员")
    return "已转接回分类员"


# ============================================
# 步骤 3：创建各角色的节点函数
# ============================================

def triage_node(state: AgentState) -> dict:
    """分类员节点 - 识别意图并决定是否转接"""
    print("\n   执行节点: triage_node")

    messages = [
        SystemMessage(content="""你是前台分类员。你的职责是识别用户意图并转接给合适的专员。
                - 如果用户要退款或查询退款状态，调用 transfer_to_refunds
                - 如果用户有技术问题，调用 transfer_to_tech_support
                - 友好地引导用户"""),
                        *state["messages"]
    ]

    tools = [transfer_to_refunds, transfer_to_tech_support]
    model_with_tools = model.bind_tools(tools)

    response = model_with_tools.invoke(messages)

    # 检查是否有工具调用
    if response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]
            for t in tools:
                if t.name == tool_name:
                    result = t.invoke({})
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                    break

        # 根据工具调用决定下一步
        tool_names = [tc["name"] for tc in response.tool_calls]
        if "transfer_to_refunds" in tool_names:
            return {"messages": [response, *tool_messages], "current_step": "refunds"}
        elif "transfer_to_tech_support" in tool_names:
            return {"messages": [response, *tool_messages], "current_step": "tech_support"}

    return {"messages": [response], "current_step": "triage"}


def refunds_node(state: AgentState) -> dict:
    """退款专员节点 - 处理退款相关问题"""
    print("\n   执行节点: refunds_node")

    messages = [
        SystemMessage(content="""你是退款专员。你专门处理退款相关问题。
                    如果用户的问题不是退款相关，调用 transfer_back_to_triage 转回分类员。
                    **重要**：请直接回答用户的问题，不要说"我需要转接"之类的话。"""),
                            *state["messages"]
                ]

    tools = [transfer_back_to_triage]
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]
            for t in tools:
                if t.name == tool_name:
                    result = t.invoke({})
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                    break

        if "transfer_back_to_triage" in [tc["name"] for tc in response.tool_calls]:
            return {"messages": [response, *tool_messages], "current_step": "triage"}

    return {"messages": [response], "current_step": "refunds"}


def tech_support_node(state: AgentState) -> dict:
    """技术支持节点 - 处理技术问题"""
    print("\n   执行节点: tech_support_node")

    messages = [
        SystemMessage(content="""你是技术支持专员。你专门处理技术问题和故障报修。
                                如果用户的问题不是技术相关，调用 transfer_back_to_triage 转回分类员。
                                **重要**：请直接回答用户的问题，不要说"我需要转接"之类的话。"""),
                                        *state["messages"]
    ]

    tools = [transfer_back_to_triage]
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]
            for t in tools:
                if t.name == tool_name:
                    result = t.invoke({})
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                    break

        if "transfer_back_to_triage" in [tc["name"] for tc in response.tool_calls]:
            return {"messages": [response, *tool_messages], "current_step": "triage"}

    return {"messages": [response], "current_step": "tech_support"}


# ============================================
# 步骤 4：构建图
# ============================================

def should_end(state: AgentState) -> str:
    """检查是否应该结束对话"""
    messages = state["messages"]
    if messages:
        last_message = messages[-1]
        # 如果是 AI 的直接回复（没有工具调用），则结束
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            return "__end__"
    return "continue"


def route_by_step(state: AgentState) -> str:
    """根据 current_step 路由到对应节点"""
    step = state.get("current_step", "triage")
    print(f"\n   路由到: {step}")
    return step


def router_node(state: AgentState) -> dict:
    """空操作路由节点"""
    return {}


# 创建图
builder = StateGraph(AgentState)

# 添加各角色节点
builder.add_node("triage", triage_node)
builder.add_node("refunds", refunds_node)
builder.add_node("tech_support", tech_support_node)

# 添加路由节点
builder.add_node("router", router_node)

# 从 START 到 router
builder.add_edge(START, "router")

# 从 router 根据 current_step 路由到对应节点
builder.add_conditional_edges(
    "router",
    route_by_step,
    {
        "triage": "triage",
        "refunds": "refunds",
        "tech_support": "tech_support",
    },
)

# 每个节点执行后检查是否结束或回到 router
for step_name in ["triage", "refunds", "tech_support"]:
    builder.add_conditional_edges(
        step_name,
        should_end,
        {
            "__end__": "__end__",
            "continue": "router",
        },
    )

# 编译图
graph = builder.compile()



def main():


    # 测试场景
    test_scenarios = [
        {
            "name": "场景 1: 退款请求",
            "messages": [
                {"role": "user", "content": "你好，我想申请退款"},
            ],
        },
        {
            "name": "场景 2: 技术支持",
            "messages": [
                {"role": "user", "content": "你好，我的 APP 崩溃了"},
            ],
        },
    ]

    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f" {scenario['name']}")
        print(f"{'='*60}")

        # 运行图
        final_state = graph.invoke({
            "messages": scenario["messages"],
            "current_step": "triage",
        })

        # 打印最终回复
        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"\nAI 回复:")
            print(f"{'-'*60}")
            print(last_message.content)
            print(f"{'-'*60}")

        print(f"\n 最终状态: current_step = {final_state.get('current_step')}")




if __name__ == "__main__":
    main()
