

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

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
# 步骤 1：创建带状态持久化的子智能体
# ============================================

# 带记忆的子智能体
memory_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是一个个人助手，负责记录用户的偏好和信息。
你会记住用户告诉你的信息，并在后续对话中使用。
""",
    checkpointer=MemorySaver(),  # 启用检查点，保持记忆
)

# 普通无状态的子智能体（对比用）
stateless_agent = create_agent(
    model=model_deepseek,
    system_prompt="""你是一个普通助手。
你不记得之前的对话。
""",
)


# ============================================
# 步骤 2：封装为工具
# ============================================

SESSION_ID = "user-session-001"  # 固定会话 ID，用于保持记忆

@tool
def call_memory_agent(query: str) -> str:
    """调用带记忆的个人助手。

    这个助手会记住你之前告诉它的信息。

    Args:
        query: 查询或要记录的信息
    """
    print(f"\n  调用记忆子智能体: {query}")
    config = {"configurable": {"thread_id": SESSION_ID}}
    result = memory_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    }, config=config)
    return result["messages"][-1].content


@tool
def call_stateless_agent(query: str) -> str:
    """调用无记忆的普通助手（对比用）。

    这个助手不记得之前的对话。

    Args:
        query: 查询
    """
    print(f"\n   调用无状态子智能体: {query}")
    result = stateless_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


# ============================================
# 步骤 3：创建主智能体
# ============================================

supervisor = create_agent(
    model=model_glm,
    tools=[call_memory_agent, call_stateless_agent],
    system_prompt="""你是一个助手协调员。

            你可以调用：
            - call_memory_agent: 带记忆的助手（会记住用户信息）
            - call_stateless_agent: 无记忆的助手（不记得之前的事）
            
            用户会告诉你一些信息，然后问你是否记得。
            用两个助手分别测试，展示记忆 vs 无记忆的区别。
            
            注意：实际调用工具。
            """,
            )


def main():

    print(f"👤 用户: 我喜欢 Python 编程，最喜欢的框架是 FastAPI")
    print(f"{'='*60}")

    result = supervisor.invoke({
        "messages": [{"role": "user", "content": "我喜欢 Python 编程，最喜欢的框架是 FastAPI"}]
    })
    print(f"{'-'*60}")
    print(result["messages"][-1].content)
    print(f"{'-'*60}")

    # 第二轮：询问记忆
    print(f"\n{'='*60}")
    print(f"用户: 你还记得我喜欢什么编程框架吗？")
    print(f"{'='*60}")

    result = supervisor.invoke({
        "messages": result["messages"] + [{"role": "user", "content": "你还记得我喜欢什么编程框架吗？"}]
    })

    print(f"{'-'*60}")
    print(result["messages"][-1].content)
    print(f"{'-'*60}")



if __name__ == "__main__":
    main()
