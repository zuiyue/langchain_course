
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

SAMPLE_DOCS = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。Python 强调代码可读性。",
        metadata={"source": "python.md"},
    ),
    Document(
        page_content="Django 是 Python 的全栈 Web 框架，适合快速开发。Flask 是轻量级微框架。",
        metadata={"source": "python_web.md"},
    ),
    Document(
        page_content="机器学习是 AI 的子领域，分为监督学习、无监督学习和强化学习。",
        metadata={"source": "ml.md"},
    ),
    Document(
        page_content="深度学习使用神经网络，TensorFlow 和 PyTorch 是主流框架。",
        metadata={"source": "dl.md"},
    ),
    Document(
        page_content="AI 伦理关注算法偏见、数据隐私和系统透明性。",
        metadata={"source": "ethics.md"},
    ),
]




embeddings = OpenAIEmbeddings(
    model="embedding-3",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL
)


vector_store = RedisVectorStore.from_documents(
    documents=SAMPLE_DOCS,
    embedding=embeddings,
    redis_url="redis://localhost:6379/0",
    index_name="redis_demo",
)

retriever=vector_store.as_retriever(search_kwargs={"k": 3})



model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)

@tool
def search_knowledeg_base(query:str)->str:
    """搜索内部知识库获取相关信息，
        当用户询问关于 python,机器学习，深度学习或者AI伦理的问题时候调用哪个该工具
        Args:
            query:搜索查询
    """
    print("search_knowledeg_base................")
    docs = vector_store.similarity_search(query,k=2)

    if not  docs:
        return "未找到相关信息"

    results =[]

    for doc in docs:
        results.append(f"[{doc.metadata.get("source")}]{doc.page_content}")

    return "\n\n".join(results)

@tool
def list_topics()->str:
    """列出知识库所有主题。
    当用户想了解知识库覆盖范围的是偶调用此工具
    :return:
    """
    print("list_topics................")
    topic=set()

    for doc in SAMPLE_DOCS:
        source = doc.metadata.get("source","unknown")
        topic.add(source.replace(".md",""))

    return  f"知识库主题：{', '.join(topic)}"


agent = create_agent(
    model=model,
    tools=[list_topics, search_knowledeg_base],
    system_prompt="""你是一个智能助手，可以访问内部知识库来回答问题。

                你可以使用以下工具：
                - search_knowledge_base(query): 搜索知识库获取相关信息
                - list_topics(): 列出知识库中所有主题
                
                工作流程：
                1. 理解用户问题
                2. 如果问题与知识库相关，调用 search_knowledge_base 搜索
                3. 如果不确定知识库覆盖范围，先调用 list_topics
                4. 基于检索结果回答用户问题
                5. 如果知识库中没有相关信息，明确告知用户
                
                 重要：请基于检索到的信息回答，不要编造不存在的内容。
                """,
)


def __main(question:str):


    user_input = {
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    result=agent.invoke(user_input)
    print(result["messages"][-1].content)



if __name__ == "__main__":
    __main("Python 有什么特点？")
    print("*"*50)
    __main("你的知识库覆盖那些主题")
    print("*" * 50)
    __main("今天天气怎么样个？")
    print("*" * 50)