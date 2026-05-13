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


research_agent=create_agent(
    model=model_deepseek,
    system_prompt="""
        你是一个专业的市场研究助手。
        你的任务是帮助用户进行市场调研、竞争分析和行业趋势分析。
        重要：在你的最后一条回复消息中，必须包含一个简洁的摘要。
        格式："摘要：[2-3 句话总结关键发现]
    """
)


@tool
def call_research_agent(query: str) -> str:
    """"
        调用市场研究助手进行市场分析和竞争情报收集。

        当你需要进行市场调研、竞争分析、行业趋势分析时使用此工具。
        不适合用于财务分析或技术评估。
    Args:
        query: 研究查询，例如："分析中国 AI 助手市场的竞争格局"
    """

    print(f"调用research_agent：{query}")

    result = research_agent.invoke({
        "messages":[{"role":"user","content":query}],
    })

    return result["messages"][-1].content



supervisor_agent=create_agent(
    model=model_glm,
    tools=[call_research_agent],
    system_prompt="""
        你是一个项目协调员（Supervisor），负责分析用户的请求并分配任务。

            你可以调用以下工具：
            - call_research_agent: 进行市场/竞争/行业研究
            
            工作流程：
            1. 分析用户需求，判断是否需要研究
            2. 如果需要，调用 call_research_agent 进行研究
            3. 整合研究结果，给用户一个完整的回复
            
            注意：不要直接说"我需要调用工具"，而是实际调用并整合结果。
    """
)


if __name__ == '__main__':
    result = supervisor_agent.invoke({
        "messages":[{"role":"user","content":"帮我分析一下中国AI助手市场的竞争格局"}],
    })

    print( result["messages"][-1].content)