
from typing import TypedDict, List

from dns.resolver import Answer
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel,Field

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

SAMPLE_DOCS = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。Python 强调代码可读性和简洁性，使用缩进划分代码块。",
        metadata={"source": "python.md"},
    ),
    Document(
        page_content="Python 广泛应用于 Web 开发、数据科学、人工智能和自动化。Django 和 Flask 是流行的 Web 框架。",
        metadata={"source": "python_apps.md"},
    ),
    Document(
        page_content="机器学习是人工智能的子领域，专注于从数据中学习的算法。监督学习使用标记数据，无监督学习发现隐藏模式。",
        metadata={"source": "ml.md"},
    ),
    Document(
        page_content="深度学习使用多层神经网络。TensorFlow 和 PyTorch 是主流的深度学习框架。",
        metadata={"source": "dl.md"},
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

retriever = vector_store.as_retriever(search_kwargs={"k": 3})



model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


class HybridRAGState(TypedDict):
    original_query: str
    rewrite_query: str
    documents: List[Document]
    answer:str
    is_valid:bool




def rewrite_query(state:HybridRAGState)->dict:

    class RewrittenQuery(BaseModel):
        query: str=Field(description="重写后的查询，更适合检索")

    query_gen =model.with_structured_output(RewrittenQuery)

    result = query_gen.invoke([SystemMessage(content="将用户查询重写为更适合向量检索的形式"),HumanMessage(content=state["original_query"])])

    print(f"Original query: {state['original_query']},Rewritten query: {result.query}")
    return {"rewrite_query":result.query}




def retrieve_document(state:HybridRAGState) -> dict:

    docs = retriever.invoke(state["rewrite_query"])

    return {"documents":docs}

def gen_answer(state:HybridRAGState)->dict:

    context = "\n\n".join(str(state["documents"]))

    prompt = f"""请基于以下参考资料回答问题。
        参考资料：
        {context}
    
        问题：{state['original_query']}
    
        如果参考资料中没有相关信息，请明确回复"未找到相关信息"。"""

    result = model.invoke(
        [SystemMessage(content="你是一个知识回答助手，请参考资料回答"), HumanMessage(content=prompt)])

    return {"answer": result.content}



def validate_answer(state:HybridRAGState)->dict:
    class ValidationResult(BaseModel):
        is_valid: bool=Field(description="答案是否基于参考资料，而非编造")
        reason: str = Field(description="校验理由")

    validator = model.with_structured_output(ValidationResult)

    result = validator.invoke(
        [SystemMessage(content="""请校验一下答案是否基于提供的参考材料,
            如果答案是编造的（不在参考资料中），返回is_valid=False.
            如果答案确实来自参考资料，返回is_valid=True
        """), HumanMessage(content=f"参考资料：\n{state['documents']}\n\n答案：{state['answer']}")])

    print(f"is_valid : {result.is_valid}")
    print(f"reason : {result.reason}")

    return {"is_valid": result.is_valid}



def handle_invalid_answer(state:HybridRAGState)->dict:
    """处理无效答案（重新检索或者回复无法回答）"""
    print("handle_invalid_answer........")
    return {
        "answer": "抱歉，基于当前的知识库，我无法找到此问题相关的准确信息",
    }

def check_validity(state:HybridRAGState)->str:
    """检查答案是否有效"""

    if state.get("is_valid", False):
        return "END"
    else:
        return "handle_invalid_answer"


builder = StateGraph(HybridRAGState)

builder.add_node("rewrite_query",rewrite_query)
builder.add_node("retrieve_document",retrieve_document)
builder.add_node("gen_answer",gen_answer)
builder.add_node("validate_answer",validate_answer)
builder.add_node("handle_invalid_answer",handle_invalid_answer)

builder.add_edge(START,"rewrite_query")
builder.add_edge("rewrite_query","retrieve_document")
builder.add_edge("retrieve_document","gen_answer")
builder.add_edge("gen_answer","validate_answer")

builder.add_conditional_edges(
    "validate_answer",
    check_validity,
    {"END":END,"handle_invalid_answer":"handle_invalid_answer"}
)


builder.add_edge("handle_invalid_answer",END)


workflow = builder.compile()




if __name__ == "__main__":
    result=workflow.invoke({
        "original_query":"Python 编程语言有哪些特点？",
        "rewrite_query":[],
        "documents":[],
        "answer":"",
        "is_valid":False,

    })

    print(result["answer"])
    print("*"*100)
    result = workflow.invoke({
        "original_query": "量子计算机的最新突破是什么？",
        "rewrite_query": [],
        "documents": [],
        "answer": "",
        "is_valid": False,

    })

    print(result["answer"])