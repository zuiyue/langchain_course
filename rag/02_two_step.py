from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

SAMPLE_DOCS = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。Python 的设计哲学强调代码可读性，使用显著的缩进来划分代码块。",
        metadata={"source": "python_intro.md", "topic": "python"},
    ),
    Document(
        page_content="Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。Python 的标准库非常丰富，被称为'自带电池'（batteries included）。",
        metadata={"source": "python_features.md", "topic": "python"},
    ),
    Document(
        page_content="机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。监督学习使用标记数据训练模型，无监督学习用于发现数据中的隐藏模式。",
        metadata={"source": "ml_intro.md", "topic": "machine_learning"},
    ),
    Document(
        page_content="深度学习是机器学习的一个分支，使用多层神经网络进行学习。常见的深度学习框架包括 TensorFlow、PyTorch 和 Keras。",
        metadata={"source": "dl_intro.md", "topic": "machine_learning"},
    ),
    Document(
        page_content="AI 伦理关注人工智能技术发展中的道德和社会问题。算法偏见和数据隐私是 AI 伦理中的核心议题。透明性和可解释性是构建可信 AI 系统的关键要求。",
        metadata={"source": "ai_ethics.md", "topic": "ai_ethics"},
    ),
]



embeddings = OpenAIEmbeddings(
    model="embedding-3",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL
)


vector_store=InMemoryVectorStore.from_documents(
    documents=SAMPLE_DOCS,
    embedding=embeddings,
)


retriever=vector_store.as_retriever(search_kwargs={"k": 3})



model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)



def __main(question:str):


    agent = create_agent(
        model=model,
        system_prompt="你是一个知识问答助手，请基于参开资料给出准确回答。"
    )

    docs=retriever.invoke(question )


    context="\n\n".join([ doc.page_content for doc in docs])


    content=f"""请基于以下参考资料回答用户的问题。
                        参考资料：
                        {context}
                        问题：{question}
                        请基于参考资料给出准确的回答。如果参考资料中没有相关信息，请明确说明。"""

    print(content)
    print("*"*50)

    user_input = {
        "messages": [
            {"role": "user", "content": content}
        ]
    }
    result=agent.invoke(user_input)
    print(result["messages"][-1].content)



if __name__ == "__main__":
    __main("python 编程语言有什么特点？")