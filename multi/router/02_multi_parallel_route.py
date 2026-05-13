
import sys
from pathlib import Path

from langchain.agents import create_agent

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL

model_deepseek = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

model_glm = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)



# ============================================
# 步骤 1：定义状态
# ============================================

class RouterState(MessagesState):
    """路由器状态，包含查询和结果集合"""
    query: str = ""
    target_agents: list = []
    expert_results: dict = {}


# ============================================
# 步骤 2：创建专门代理
# ============================================

# 市场分析专家
market_agent = create_agent(
    model=model_deepseek,
    tools=[],
    system_prompt="""你是市场分析专家。
你专门分析市场规模、竞争格局和行业趋势。
请提供简洁、专业的市场分析，包含关键数据和洞察。
""",
)

# 技术评估专家
tech_agent = create_agent(
    model=model_deepseek,
    tools=[],
    system_prompt="""你是技术评估专家。
你专门评估技术可行性、技术栈选择和实现难度。
请提供简洁、专业的技术评估，包含关键建议和风险提示。
""",
)

# 商业分析专家
business_agent = create_agent(
    model=model_deepseek,
    tools=[],
    system_prompt="""你是商业分析专家。
你专门分析商业模式、收入预测和盈利可行性。
请提供简洁、专业的商业分析，包含关键财务指标。
""",
)


# ============================================
# 步骤 3：定义路由和聚合函数
# ============================================

def classify_node(state: RouterState) -> dict:
    """分类节点：识别查询需要哪些专家，存储到状态中"""
    query = state.get("query", "").lower()

    agents_needed = []
    if any(kw in query for kw in ["市场", "竞争", "行业", "趋势", "用户"]):
        agents_needed.append("market")
    if any(kw in query for kw in ["技术", "开发", "实现", "架构", "性能"]):
        agents_needed.append("tech")
    if any(kw in query for kw in ["商业", "收入", "盈利", "投资", "成本"]):
        agents_needed.append("business")
    if not agents_needed:
        agents_needed = ["market", "tech", "business"]

    print(f"\n 需要专家: {', '.join(agents_needed)}")
    return {"target_agents": agents_needed}


def route_to_experts(state: RouterState):
    """条件边函数：根据 target_agents 生成 Send 列表，实现并行扇出"""
    agents_needed = state.get("target_agents", ["market", "tech", "business"])
    return [
        Send(
            f"{agent}_expert",
            {"messages": [HumanMessage(content=state["query"])], "query": state["query"]},
        )
        for agent in agents_needed
    ]


def synthesize_results(state: RouterState) -> dict:
    """合成函数：整合各专家的结果"""
    print("\n 合成各专家结果...")

    # 这里简化处理，实际应该用 LLM 来整合
    messages = state.get("messages", [])

    # 生成整合提示
    synthesis_prompt = """请整合以下各专家的分析结果，给出一份综合评估报告：

"""
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            synthesis_prompt += f"\n专家 {i+1} 分析:\n{msg.content}\n"

    synthesis_prompt += "\n请按照以下格式输出综合评估：\n"
    synthesis_prompt += "1. 市场评估\n2. 技术评估\n3. 商业评估\n4. 综合建议\n"

    # 调用 LLM 进行整合
    model = init_chat_model(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    )

    response = model.invoke([
        SystemMessage(content="你是综合评估专家，负责整合各垂直领域的分析结果。"),
        HumanMessage(content=synthesis_prompt),
    ])

    return {"messages": [response]}


# ============================================
# 步骤 4：构建图
# ============================================

def should_end(state: RouterState) -> str:
    """检查是否结束"""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return "__end__"
    return "continue"


# 创建图
builder = StateGraph(RouterState)

# 添加路由节点（并行扇出）
builder.add_node("classify", classify_node)
builder.add_edge(START, "classify")

builder.add_node("market_expert", market_agent)
builder.add_node("tech_expert", tech_agent)
builder.add_node("business_expert", business_agent)

builder.add_conditional_edges(
    "classify",
    route_to_experts,
    ["market_expert", "tech_expert", "business_expert"],
)

# 各专家完成后进入合成节点
for expert in ["market_expert", "tech_expert", "business_expert"]:
    builder.add_edge(expert, "synthesize")

# 添加合成节点
builder.add_node("synthesize", synthesize_results)
builder.add_conditional_edges("synthesize", should_end, ["__end__", "synthesize"])

graph = builder.compile()


# ============================================
# 测试运行
# ============================================

def main():


    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 需要市场+技术+商业分析",
            "query": "我想开发一个 AI 助手产品，帮我评估市场前景、技术可行性和商业价值",
        },
        {
            "name": "场景 2: 只需要市场分析",
            "query": "分析一下中国 AI 助手市场的竞争格局和行业趋势",
        },
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f" {test_case['name']}")
        print(f"{'='*60}")
        print(f"👤 用户: {test_case['query']}")

        # 运行图
        final_state = graph.invoke({
            "messages": [HumanMessage(content=test_case["query"])],
            "query": test_case["query"],
            "target_agents": [],
            "expert_results": {},
        })

        # 打印最终回复
        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"\n 综合评估报告:")
            print(f"{'='*60}")
            print(last_message.content)
            print(f"{'='*60}")




if __name__ == "__main__":
    main()