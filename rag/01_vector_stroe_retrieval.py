from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

SAMPLE_DOCS = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。",
        metadata={"source": "python_intro.md", "topic": "python", "year": 1991},
    ),
    Document(
        page_content="Python 支持多种编程范式，包括面向对象、函数式和过程式编程。",
        metadata={"source": "python_features.md", "topic": "python"},
    ),
    Document(
        page_content="机器学习是人工智能的子领域，专注于从数据中学习的算法。",
        metadata={"source": "ml_intro.md", "topic": "machine_learning"},
    ),
    Document(
        page_content="深度学习使用多层神经网络，是机器学习的一个重要分支。",
        metadata={"source": "dl_intro.md", "topic": "machine_learning"},
    ),
    Document(
        page_content="TensorFlow 和 PyTorch 是最流行的深度学习框架。",
        metadata={"source": "dl_frameworks.md", "topic": "machine_learning"},
    ),
    Document(
        page_content="AI 伦理关注算法偏见、数据隐私和系统透明性。",
        metadata={"source": "ai_ethics.md", "topic": "ai_ethics"},
    ),
    Document(
        page_content="自然语言处理（NLP）是 AI 的一个重要领域，涉及文本理解和生成。",
        metadata={"source": "nlp_intro.md", "topic": "nlp"},
    ),
    Document(
        page_content="计算机视觉是 AI 的另一个重要领域，涉及图像识别和理解。",
        metadata={"source": "cv_intro.md", "topic": "computer_vision"},
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


result = vector_store.similarity_search("Python 编程语言的特点",k=3)
for i,doc in enumerate(result,1):
    print(f"{i}. {doc.metadata}")
    print(f"{i}. {doc.page_content}")


print("*"*100)
result = vector_store.similarity_search_with_score("深度学习和神经网络",k=3)
for i,(doc,score) in enumerate(result,1):
    print(f"{i}. {doc.metadata}")
    print(f"{i}. {score}")
    print(f"{i}. {doc.page_content}")

print("*"*100)
result = vector_store.max_marginal_relevance_search("AI 和机器学习",k=3)
for i,doc in enumerate(result,1):
    print(f"{i}. {doc.metadata}")
    print(f"{i}. {doc.page_content}")

print("*"*100)
def topic_filter(doc:Document)->bool:
    print("******",doc.metadata.get("topic"))
    return doc.metadata.get("topic") == "machine_learning"

result = vector_store.similarity_search("编程",k=3,filter=topic_filter)
for i,doc in enumerate(result,1):
    print(f"{i}. {doc.metadata}")
    print(f"{i}. {doc.page_content}")