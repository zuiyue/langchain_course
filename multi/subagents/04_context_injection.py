
import sys
from pathlib import Path


from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime


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
# 步骤 1：创建子智能体
# ============================================

research_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是市场研究专家。
                你会收到用户的查询以及之前的对话历史。
                请结合上下文给出更精准的研究结果。
                必须在最后一条消息中包含摘要。
                格式："摘要：[2-3 句话总结]"
                """,
                )


# ============================================
# 步骤 2：封装工具（带上下文注入）
# ============================================

@tool
def call_research_with_context(query: str, runtime: ToolRuntime) -> str:
    """调用市场研究专家，并带入之前的对话历史。

    当你需要基于之前讨论内容进行研究时使用此工具。

    Args:
        query: 研究查询
        runtime: 工具运行时上下文（可访问主智能体状态）
    """
    print(f"\n  调用研究子智能体（带上下文）: {query}")

    # 从主智能体状态中提取对话历史
    main_messages = runtime.state.get("messages", [])

    # 过滤：只提取 HumanMessage 和 AIMessage（排除 ToolMessage 和带 tool_calls 的消息）
    filtered_messages = []
    for msg in main_messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            # 只提取纯文本的 AI 回复（没有工具调用）
            filtered_messages.append(msg)

    # 提取最近 3 条过滤后的消息作为上下文
    recent_messages = filtered_messages[-3:] if len(filtered_messages) > 3 else filtered_messages

    # 构造子智能体输入：包含历史 + 当前查询
    sub_messages = [
        SystemMessage(content="以下是之前的对话历史，供你参考："),
        *recent_messages,
        HumanMessage(content=query),
    ]

    print(f" 传递给子智能体的消息数量: {len(sub_messages)}")

    result = research_agent.invoke({
        "messages": sub_messages
    })
    return result["messages"][-1].content


# ============================================
# 步骤 3：创建主智能体
# ============================================

supervisor = create_agent(
    model=model_glm,
    tools=[call_research_with_context],
    system_prompt="""你是一个项目协调员。
                你可以调用 call_research_with_context(query, runtime) 进行研究。
                此工具会自动带入之前的对话历史。
                
                """,
                )


if __name__ == "__main__":
    result = supervisor.invoke({
        "messages": [HumanMessage(content="我想分析一下 AI 助手市场")]
    })

    assistant_message = result["messages"][-1].content

    print(assistant_message)
