

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
from langgraph.types import Command
from pydantic import BaseModel, Field

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
    """路由器状态"""
    query: str = ""
    category: str = ""


# ============================================
# 步骤 2：创建 LLM 分类器
# ============================================

class QueryCategory(BaseModel):
    """查询分类模型"""
    category: Literal["technical", "creative", "analytical"] = Field(
        description="查询的分类: technical=技术问题, creative=创意写作, analytical=数据分析"
    )
    confidence: float = Field(
        description="分类置信度 0-1"
    )
    reason: str = Field(
        description="分类理由"
    )


# 创建分类模型
classifier_model = init_chat_model(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
).with_structured_output(QueryCategory)


def classify_with_llm(query: str) -> str:
    """使用 LLM 分类查询"""
    print(f"\n 正在使用 LLM 分类查询...")

    result = classifier_model.invoke([
        SystemMessage(content="""你是一个查询分类器。请将用户查询分类为以下类别：
                - technical: 技术问题、编程、代码、调试等
                - creative: 创意写作、故事、诗歌、文案等
                - analytical: 数据分析、统计、图表、报告等
                
                请返回分类结果、置信度和理由。"""),
                        HumanMessage(content=query),
                    ])

    print(f"   分类结果: {result.category} (置信度: {result.confidence:.2f})")
    print(f"   理由: {result.reason}")

    return result.category


# ============================================
# 步骤 3：创建专门代理
# ============================================

# 技术专家
technical_agent = create_agent(
    model=model_deepseek,
    tools=[],
    system_prompt="""你是技术专家。
            你专门解决编程问题、代码调试和技术架构。
            请提供具体的代码示例和技术建议。
            """,
            )

# 创意写作专家
creative_agent = create_agent(
    model=model_deepseek,
    tools=[],
    system_prompt="""你是创意写作专家。
            你擅长写故事、诗歌、文案和创意内容。
            请提供富有创意和感染力的内容。
            """,
            )

# 数据分析专家
analytical_agent = create_agent(
    model=model_deepseek,
    tools=[],
    system_prompt="""你是数据分析专家。
            你擅长统计分析、数据可视化和报告撰写。
            请提供数据驱动的分析和具体建议。
            """,
            )


# ============================================
# 步骤 4：定义路由函数
# ============================================

def route_to_specialist(state: RouterState) -> Command:
    """路由函数：使用 LLM 分类，然后路由到对应专家"""
    query = state.get("query", "")
    category = classify_with_llm(query)

    # 更新状态中的分类结果
    state["category"] = category

    # 映射分类到专家节点
    agent_map = {
        "technical": "technical_expert",
        "creative": "creative_expert",
        "analytical": "analytical_expert",
    }

    target = agent_map.get(category, "technical_expert")
    print(f"\n 路由到: {target}")

    return Command(goto=target)


# ============================================
# 步骤 5：构建图
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
builder.add_node("route_to_specialist", route_to_specialist)
builder.add_edge(START, "route_to_specialist")

# 添加专家节点
builder.add_node("technical_expert", technical_agent)
builder.add_node("creative_expert", creative_agent)
builder.add_node("analytical_expert", analytical_agent)

# 从路由节点条件边到各专家
builder.add_conditional_edges(
    "route_to_specialist",
    lambda state: {
        "technical": "technical_expert",
        "creative": "creative_expert",
        "analytical": "analytical_expert",
    }.get(state.get("category", ""), "technical_expert"),
    ["technical_expert", "creative_expert", "analytical_expert"],
)

# 各专家执行后结束
for expert in ["technical_expert", "creative_expert", "analytical_expert"]:
    builder.add_conditional_edges(expert, should_end, ["__end__", expert])

graph = builder.compile()


# ============================================
# 测试运行
# ============================================

def main():


    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 技术问题",
            "query": "Python 中如何优化这个函数的性能？def process_data(items): return [x*2 for x in items]",
        },
        {
            "name": "场景 2: 创意写作",
            "query": "帮我写一首关于春天和希望的短诗",
        },
        {
            "name": "场景 3: 数据分析",
            "query": "帮我分析一下这组销售数据的趋势：1 月 100 万，2 月 120 万，3 月 150 万，4 月 180 万",
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
            "category": "",
        })

        # 打印最终回复
        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"\n 专家回复:")
            print(f"{'-'*60}")
            print(last_message.content[:300])
            print(f"{'-'*60}")

        print(f"\n最终分类: {final_state.get('category')}")




if __name__ == "__main__":
    main()
