from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class KeywordRetriever(BaseRetriever):
    docs:List[Document]

    def _get_relevant_documents(self, query: str ):
        keyword=query.lower()

        return [ doc for doc in self.docs if keyword in doc.page_content.lower()]



if __name__ == "__main__":
    retriever=KeywordRetriever(
        docs=[
             Document(page_content="LangGraph 很好用"),
             Document(page_content="LangChain 也很好用"),
             Document(page_content="DeepAgent 更好用 "),
        ]
    )

    retriever_tool= retriever.as_tool(
        name="doc_search",
        description="根据关键字检索文档",
    )

    result =retriever_tool.invoke({"query":"好用"})
    print(result)


