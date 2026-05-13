

import sys
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# ============================================
# 步骤 1：定义状态
# ============================================

class RouterState(MessagesState):
    """路由器状态，包含当前目标代理标识"""
    query: str = ""
    target_agent: Literal["math_expert", "writing_expert", "coding_expert"] = ""


# ============================================
# 步骤 2：创建专门代理
# ============================================

# 数学专家
math_agent = create_agent(
    model=init_chat_model(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    ),
    tools=[],
    system_prompt="""你是数学专家。
        你专门处理数学问题、公式推导和计算。
        请直接回答用户的数学问题，给出详细的解题步骤。
        """,
)

# 写作专家
writing_agent = create_agent(
    model=init_chat_model(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    ),
    tools=[],
    system_prompt="""你是写作专家。
        你专门处理文章撰写、文案创作和文案优化。
        请直接回答用户的写作需求，提供具体的建议和示例。
        """,
)

# 编程专家
coding_agent = create_agent(
    model=init_chat_model(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    ),
    tools=[],
    system_prompt="""你是编程专家。
        你专门处理代码编写、调试和技术问题解答。
        请直接回答用户的编程问题，提供代码示例和解释。
        """,
)


# ============================================
# 步骤 3：定义路由函数
# ============================================

def classify_query(query: str) -> str:
    """简单的规则分类器（实际场景可用 LLM 分类）"""
    query_lower = query.lower()

    # 数学关键词
    math_keywords = ["数学", "计算", "公式", "几何", "代数", "概率", "三角", "积分", "微分", "等于", "多少"]
    # 写作关键词
    writing_keywords = ["写", "文章", "文案", "报告", "总结", "摘要", "翻译", "润色", "改写"]
    # 编程关键词
    coding_keywords = ["代码", "编程", "python", "java", "javascript", "函数", "调试", "bug", "错误", "算法"]

    # 关键词匹配
    for keyword in math_keywords:
        if keyword in query_lower:
            return "math_expert"

    for keyword in writing_keywords:
        if keyword in query_lower:
            return "writing_expert"

    for keyword in coding_keywords:
        if keyword in query_lower:
            return "coding_expert"

    # 默认路由到写作专家
    return "writing_expert"


def route_to_expert(state: RouterState) -> Command:
    """路由函数：根据查询分类，路由到对应的专家"""
    query = state.get("query", "")
    target = classify_query(query)

    print(f"\n 分类结果: {target}")

    return Command(goto=target)


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

# 添加路由节点
builder.add_node("route_to_expert", route_to_expert)

builder.add_edge(START, "route_to_expert")

# 添加专家节点
builder.add_node("math_expert", math_agent)
builder.add_node("writing_expert", writing_agent)
builder.add_node("coding_expert", coding_agent)

# 从路由节点条件边到各专家
builder.add_conditional_edges(
    "route_to_expert",
    lambda state: state.get("target_agent") or classify_query(state.get("query", "")),
    ["math_expert", "writing_expert", "coding_expert"],
)

# 每个专家节点执行后结束
for expert in ["math_expert", "writing_expert", "coding_expert"]:
    builder.add_conditional_edges(expert, should_end, ["__end__", expert])

graph = builder.compile()


# ============================================
# 测试运行
# ============================================

def main():

    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 数学问题",
            "query": "请帮我计算三角形的面积，底是 10，高是 5",
        },
        {
            "name": "场景 2: 写作需求",
            "query": "帮我写一段关于 AI 发展趋势的文章摘要",
        },
        {
            "name": "场景 3: 编程问题",
            "query": "Python 中如何实现列表去重？请给出代码",
        },
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f" {test_case['name']}")
        print(f"{'='*60}")
        print(f" 用户: {test_case['query']}")

        # 运行图
        final_state = graph.invoke({
            "messages": [HumanMessage(content=test_case["query"])],
            "query": test_case["query"],

        })

        # 打印最终回复
        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"\n 专家回复:")
            print(f"{'-'*60}")
            print(last_message.content[:300])
            print(f"{'-'*60}")



if __name__ == "__main__":
    main()
