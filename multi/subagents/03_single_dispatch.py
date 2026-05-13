

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
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
# 步骤 1：创建多个子智能体并维护注册表
# ============================================

# 市场研究子智能体
research_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是市场研究专家。
                    必须在最后一条消息中包含摘要。
                    格式："摘要：[2-3 句话总结]"
                    """,
                    )

# 写作子智能体
writer_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是专业写作者，擅长撰写文章、报告和文案。
                    必须在最后一条消息中包含你撰写的内容。
                    """,
                    )

# 审查子智能体
reviewer_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是内容审查专家，擅长检查文章质量、逻辑和语法。
                    必须在最后一条消息中包含审查意见和建议。
                    格式："摘要：[审查结果和改进建议]"
                    """,
                    )

# 翻译子智能体
translator_agent = create_agent(
    model=init_chat_model(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    ),
    system_prompt="""你是专业翻译家，擅长中英文互译。
                必须在最后一条消息中包含翻译结果。
                """,
                )

# 子智能体注册表
SUBAGENTS = {
    "research": {
        "agent": research_agent,
        "description": "市场研究、竞争分析、行业趋势",
    },
    "writer": {
        "agent": writer_agent,
        "description": "撰写文章、报告、文案",
    },
    "reviewer": {
        "agent": reviewer_agent,
        "description": "内容审查、质量检查、改进建议",
    },
    "translator": {
        "agent": translator_agent,
        "description": "中英文翻译",
    },
}


# ============================================
# 步骤 2：定义单一调度工具
# ============================================

@tool
def task(agent_name: str, query: str) -> str:
    """调用指定的子智能体执行任务。

    根据子智能体名称动态路由任务到对应的专家。

    Args:
        agent_name: 子智能体名称（research/writer/reviewer/translator）
        query: 任务描述或查询
    """
    print(f"\n  调度任务: agent={agent_name}, query={query}")

    if agent_name not in SUBAGENTS:
        available = list(SUBAGENTS.keys())
        return f"错误：未知的子智能体 '{agent_name}'。可用的有：{available}"

    agent = SUBAGENTS[agent_name]["agent"]
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


@tool
def list_agents() -> str:
    """列出所有可用的子智能体及其描述。

    当你不知道有哪些子智能体可用时调用此工具。
    """
    print("\n  列出可用子智能体:")
    agents_info = []
    for name, info in SUBAGENTS.items():
        agents_info.append(f"  - {name}: {info['description']}")
    return "\n".join(agents_info)


# ============================================
# 步骤 3：创建主智能体（Supervisor）
# ============================================

supervisor = create_agent(
    model=model_glm,
    tools=[task, list_agents],
    system_prompt="""你是一个内容创作团队的协调员（Supervisor）。

                    你可以使用以下工具：
                    - list_agents(): 查看可用的子智能体
                    - task(agent_name, query): 调用指定的子智能体执行任务
                    
                    可用的子智能体有：
                    - research: 市场研究
                    - writer: 内容创作
                    - reviewer: 内容审查
                    - translator: 翻译
                    
                    工作流程：
                    1. 分析用户需求
                    2. 如需了解可用智能体，调用 list_agents()
                    3. 使用 task(agent_name, query) 调用相应专家
                    4. 整合结果给用户
                    
                     注意：实际调用工具并整合结果。
                    """,
                    )



if __name__ == "__main__":
    result = supervisor.invoke({
        "messages": [{"role": "user", "content": "帮我写一篇关于 AI 发展趋势的文章，先做一下市场调研"}]
    })

    assistant_message = result["messages"][-1].content

    print(assistant_message)
