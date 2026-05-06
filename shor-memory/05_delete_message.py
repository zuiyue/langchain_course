from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


@before_model
def delete_message(state:AgentState,_runtime:Runtime) -> dict[str, Any] |None:
    """删除早期消息"""

    messages = state["messages"]

    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

    return None


checkpointer=InMemorySaver()

agent = create_agent(
    model=model,
    checkpointer=checkpointer,
     # middleware=[delete_message]
)

config={
    "configurable":{
        "thread_id":"thread_001"
    }
}


if __name__ == "__main__":
    questions=[
        "我叫张三，记住我的名字",
        "写一首春天的诗",
        "写一首大海的诗",
        "我叫什么",
        "我叫什么",
    ]

    for question in questions:
        result = agent.invoke({"messages":[{"content":question,"role":"user"}]},config=config)
        print(len(result["messages"]))
        print(result["messages"][-1])
