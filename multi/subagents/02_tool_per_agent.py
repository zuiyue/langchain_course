
import sys
from pathlib import Path



project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
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
# 步骤 1：创建多个专业子智能体
# ============================================

# 市场研究子智能体
research_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是市场研究专家，擅长行业分析、竞争格局和市场趋势判断。
                    必须在最后一条消息中包含关键发现的摘要。
                    格式："摘要：[2-3 句话总结]"
                    """,
                    )

# 财务分析子智能体
finance_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是财务分析专家，擅长财务数据解读、盈利能力分析和财务预测。
                    必须在最后一条消息中包含关键财务指标的摘要。
                    格式："摘要：[2-3 句话总结关键财务发现]"
                    """,
                    )

# 技术评估子智能体
tech_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是技术评估专家，擅长技术可行性分析、技术选型和架构评估。
                     必须在最后一条消息中包含关键技术建议的摘要。
                    格式："摘要：[2-3 句话总结技术建议]"
                    """,
                    )


# ============================================
# 步骤 2：将每个子智能体封装为独立工具
# ============================================

@tool
def call_research_agent(query: str) -> str:
    """调用市场研究专家进行市场分析。

    当你需要了解行业趋势、竞争格局、市场份额或市场机会时使用此工具。
    不适合用于财务分析、技术评估或法律审查。

    Args:
        query: 研究查询，例如："分析 AI 助手市场的竞争格局"
    """
    print(f"\n 调用市场研究子智能体: {query}")
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


@tool
def call_finance_agent(query: str) -> str:
    """调用财务分析专家进行财务分析。

    当你需要财务数据解读、盈利能力分析、成本分析或财务预测时使用此工具。
    不适合用于市场研究、技术评估或法律审查。

    Args:
        query: 财务查询，例如："分析这家公司的盈利能力"
    """
    print(f"\n 调用财务分析子智能体: {query}")
    result = finance_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


@tool
def call_tech_agent(query: str) -> str:
    """调用技术评估专家进行技术分析。

    当你需要技术可行性分析、技术选型建议、架构评估或性能分析时使用此工具。
    不适合用于市场研究、财务分析或法律审查。

    Args:
        query: 技术查询，例如："评估使用大模型的技术可行性"
    """
    print(f"\n 调用技术评估子智能体: {query}")
    result = tech_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


# ============================================
# 步骤 3：创建主智能体（Supervisor）
# ============================================

supervisor = create_agent(
    model=model_glm,
    tools=[call_research_agent, call_finance_agent, call_tech_agent],
    system_prompt="""你是一个项目协调员（Supervisor），负责分析用户的复杂请求并分配任务给专家团队。

                    你可以调用以下工具：
                    - call_research_agent: 市场/竞争/行业研究
                    - call_finance_agent: 财务分析/数据解读/财务预测
                    - call_tech_agent: 技术可行性/技术选型/架构评估
                    
                    工作流程：
                    1. 分析用户需求，判断需要哪些专家
                    2. 依次调用相关专家的工具
                    3. 整合所有专家的结果
                    4. 给用户一个完整的综合回复
                    
                    注意：
                    - 实际调用工具并整合结果，不要只说"我需要..."
                    - 每个专家的结果都要整合到你的最终回复中
                    """,
                    )




if __name__ == "__main__":
    result = supervisor.invoke({
        "messages": [{"role": "user", "content": "我想开发一个 AI 助手产品，帮我分析一下市场前景和技术可行性"}]
    })

    assistant_message = result["messages"][-1].content
    print(assistant_message)
