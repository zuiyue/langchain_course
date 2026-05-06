from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)
model_summarizer = init_chat_model(
    model="deepseek:deepseek-reasoner",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


summarizer = SummarizationMiddleware(
    model=model_summarizer,
    trigger=("tokens",500),
    keep=("messages",6)
)

checkpointer=InMemorySaver()

agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    middleware=[summarizer]
)

config={
    "configurable":{
        "thread_id":"thread_001"
    }
}


if __name__ == "__main__":
    questions=[
        "我叫张三，我在学习lanngchian",
        "总结一下长期记忆和短期记忆的区别",
        "句两个实际业务中的例子",
        "如果对话很长，会出现什么问题",
        "如何使用 SummarizationMiddleware 来处理这些问题",
        "我叫什么？",
    ]

    for question in questions:
        result = agent.invoke({"messages":[{"content":question,"role":"user"}]},config=config)
        print(len(result["messages"]))
        print(result["messages"][-1])
