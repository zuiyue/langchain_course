# LangChain Retrieval 核心概念


---

## 一、什么是 Retrieval？

### 1.1 从"闭卷考试"到"开卷考试"

想象你在参加考试：

```
没有 Retrieval（闭卷考试）:
  考生（LLM）只能依靠大脑中已有的知识回答问题
  
  问题：
  ❌ 知识是训练时冻结的，不知道最新信息
  ❌ 上下文窗口有限，无法一次性记住整个语料库
  ❌ 可能产生幻觉，编造不存在的信息

有 Retrieval（开卷考试）:
  考生（LLM）可以翻阅参考书和笔记
  
  流程：
  1. 看到题目（用户查询）
  2. 翻阅参考书找到相关内容（检索）
  3. 基于参考书内容回答问题（生成）
  
  优势：
  ✅ 可以获取最新信息
  ✅ 答案基于真实文档，减少幻觉
  ✅ 可以引用来源，增加可信度
```

**核心定义**：Retrieval（检索）是指在查询时**动态获取相关外部知识**的过程。它是**检索增强生成（RAG）**的基础架构，通过为 LLM 提供上下文特定的实时信息，弥补其内部知识的不足。

---

## 二、为什么需要 Retrieval？

### 2.1 LLM 的两大固有局限

```
┌─────────────────────────────────────────────────────────┐
│              LLM 的两大局限                                │
│                                                         │
│  1. 上下文有限（Context Window Limitation）               │
│     无法一次性摄入整个语料库或长文档                       │
│     即使窗口很大，长上下文中模型注意力也会分散              │
│                                                         │
│  2. 知识静态（Static Knowledge）                          │
│     训练数据在特定时间点已冻结                             │
│     缺乏训练之后的最新信息                                │
│     不知道公司内部文档、用户私有数据等                     │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Retrieval 如何解决这些问题

```
Retrieval 在运行时按需拉取数据，直接解决上述问题：

局限 1：上下文有限
  解决：不把所有文档塞给模型，而是只检索与查询最相关的片段
  类比：不是把整本百科全书给考生，而是帮他翻到相关页面

局限 2：知识静态
  解决：从实时数据源（数据库、API、文档系统）检索最新信息
  类比：考生可以查阅今天的报纸，而不是只靠记忆
```

---

## 三、Retrieval 的工作原理和架构

### 3.1 完整检索管线

以下是 RAG 的完整模块化流水线：

```
外部数据源（PDF、网页、数据库...）
  │
  ▼
┌──────────────┐
│ 文档加载器    │  从各种来源摄取数据，返回标准化 Document 对象
│ (Loader)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 文本分割器    │  将大文档拆分为小文本块（Chunks）
│ (Splitter)   │  适配模型上下文窗口，提升检索粒度
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 嵌入模型      │  将文本转换为数值向量
│ (Embedding)  │  语义相近的文本在向量空间中距离更近
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 向量数据库    │  高效存储和检索嵌入向量
│ (VectorStore)│
└──────┬───────┘
       │
       ▼ （查询时）
┌──────────────┐
│ 检索器        │  接收非结构化查询，返回相关文档
│ (Retriever)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    LLM       │  基于检索到的文档生成最终答案
│  (生成)      │
└──────┬───────┘
       │
       ▼
    用户答案
```

### 3.2 两阶段：索引 vs 查询

```
┌─────────────────────────────────────────────────────────┐
│              Retrieval 两阶段                             │
│                                                         │
│  阶段一：索引（Indexing）—— 一次性构建                    │
│  ──────────────────────────────                          │
│  文档 → 加载 → 分割 → 嵌入 → 存入向量库                   │
│                                                         │
│  阶段二：查询（Querying）—— 每次请求执行                   │
│  ──────────────────────────────                          │
│  用户问题 → 嵌入 → 向量相似度匹配 → 检索相关文档 → 生成答案 │
└─────────────────────────────────────────────────────────┘
```

---

## 四、Retrieval 的核心组件

### 4.1 五大核心组件

```
┌─────────────────────────────────────────────────────────┐
│              Retrieval 五大核心组件                        │
│                                                         │
│  1. Document Loaders（文档加载器）                        │
│     从外部源摄取数据，返回标准化的 Document 对象            │
│                                                         │
│  2. Text Splitters（文本分割器）                          │
│     将大文档拆分为独立的小文本块                            │
│                                                         │
│  3. Embedding Models（嵌入模型）                          │
│     将文本转换为数值向量                                   │
│                                                         │
│  4. Vector Stores（向量存储）                             │
│     专门用于高效存储和检索嵌入向量的数据库                  │
│                                                         │
│  5. Retrievers（检索器）                                  │
│     接收非结构化查询并返回相关文档的标准化接口              │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Document Loaders（文档加载器）

**作用**：从各种外部源摄取数据，返回标准化的 `Document` 对象。

```
支持的来源：
  ├── 本地文件：PDF、TXT、MD、CSV、JSON、Word、Excel...
  ├── 网络资源：网页、URL、API 响应
  ├── 云端存储：Google Drive、OneDrive、S3...
  ├── 协作平台：Notion、Slack、Confluence、GitHub...
  └── 数据库：SQL、MongoDB、Redis...

标准化输出：
  Document(
      page_content="文档的文本内容...",
      metadata={"source": "file.pdf", "page": 1, ...}
  )
```

### 4.3 Text Splitters（文本分割器）

**作用**：将大文档拆分为独立的小文本块（Chunks），确保适配模型上下文窗口并提升检索粒度。

```
为什么要分割？
  ❌ 整个文档作为一个单元：
     - 超出上下文窗口
     - 检索时整篇返回，包含大量无关信息
     - LLM 在长文本中难以聚焦
  
  ✅ 分割为小块：
     - 每块适配上下文窗口
     - 检索时返回最相关的片段
     - LLM 基于精准信息回答

分割策略：
  ├── 按字符数分割：每块 500 字符，重叠 50 字符
  ├── 按段落分割：保持段落完整性
  ├── 按语义分割：根据语义边界切分
  └── 递归字符分割：尝试多种分隔符，从大到小
```

### 4.4 Embedding Models（嵌入模型）

**作用**：将文本转换为数值向量，使语义相近的文本在向量空间中距离更近。

```
嵌入的本质：
  "今天天气很好"  ──►  [0.23, -0.45, 0.67, ...]  ← 1536 维向量
  "天气真不错"    ──►  [0.22, -0.44, 0.68, ...]  ← 距离很近
  "我喜欢编程"    ──►  [0.89, 0.12, -0.33, ...]  ← 距离很远

为什么用向量？
  - 计算机无法直接理解"语义相似度"
  - 但它可以计算两个向量之间的数学距离
  - 向量越近 → 语义越相似

常用嵌入模型：
  ├── OpenAI: text-embedding-3-small / large
  ├── Cohere: embed-v3
  ├── Google: text-embedding-004
  └── 开源: BGE、GTE、E5...
```

### 4.5 Vector Stores（向量存储）

**作用**：专门用于高效存储和检索嵌入向量的数据库。

```
向量数据库的核心能力：
  1. 存储：保存文本块及其嵌入向量
  2. 索引：建立高效检索结构（如 HNSW、IVF）
  3. 搜索：根据查询向量找到最近的文档向量

常见的向量数据库：
  ├── 专用型：Pinecone、Weaviate、Milvus、Qdrant
  ├── 开源型：Chroma、FAISS（Facebook）
  ├── 传统数据库扩展：PostgreSQL（pgvector）、MongoDB、Redis
  └── 云服务商：Azure AI Search、Google Vertex Matching Engine
```

### 4.6 Retrievers（检索器）

**作用**：接收非结构化查询并返回相关文档的标准化接口。

```
检索器 vs 向量数据库：
  - 向量数据库是底层存储
  - 检索器是面向应用的接口
  
  检索器封装了底层搜索逻辑，对上提供统一接口

检索器的输入输出：
  输入："北京今天的天气怎么样？"  ← 自然语言查询
  输出：[Document1, Document2, Document3]  ← 相关文档列表
```

---

## 五、三种 RAG 架构模式

LangChain 的 Retrieval 主要分为三种实现模式：

### 5.1 模式对比一览

```
┌─────────────────────────────────────────────────────────┐
│              三种 RAG 架构对比                             │
│                                                         │
│  2-Step RAG          Agentic RAG        Hybrid RAG      │
│  （两步检索）         （智能体检索）       （混合检索）      │
│                                                         │
│  固定流水线           智能体驱动          混合迭代验证      │
│  线性执行            动态决策            多轮验证          │
│  延迟可预测           延迟动态变化        延迟较高          │
│  控制力强             灵活性高            准确度高          │
│                                                         │
│  适用：FAQ、客服     适用：研究助手      适用：专业问答    │
└─────────────────────────────────────────────────────────┘
```

### 5.2 2-Step RAG（两步检索）

**流程**：严格的"先检索、后生成"，无分支判断。

```
用户提问
  │
  ▼
检索文档（固定步骤）
  │
  ▼
生成答案（固定步骤）
  │
  ▼
返回用户
```

**特点**：

| 特性 | 说明 |
|------|------|
| **架构** | 线性流水线，无条件分支 |
| **延迟** | LLM 调用次数固定，推理延迟可预测 |
| **控制力** | 强，流程完全可控 |
| **灵活性** | 低，无法处理复杂场景 |
| **适用场景** | FAQ、文档客服机器人 |

**示例**：

```python
# 2-Step RAG 伪代码
def two_step_rag(query):
    # 步骤 1：检索
    docs = retriever.invoke(query)
    context = "\n".join(doc.content for doc in docs)
    
    # 步骤 2：生成
    prompt = f"基于以下信息回答问题：\n\n{context}\n\n问题：{query}"
    answer = llm.invoke(prompt)
    
    return answer
```

### 5.3 Agentic RAG（智能体检索）

**流程**：LLM 智能体在推理链中自主判断"何时"及"如何"调用检索工具。

```
用户提问
  │
  ▼
┌──────────────┐
│  Agent 推理   │  需要外部信息吗？
└──────┬───────┘
       │
  ┌────┴────┐
  │ 需要    │ 不需要
  ▼         ▼
调用检索工具  直接回答
  │
  ▼
评估信息充足吗？
  │
  ├── 不足 ──► 重新检索或换其他来源
  │
  └── 充足 ──► 生成答案
               │
               ▼
            返回用户
```

**特点**：

| 特性 | 说明 |
|------|------|
| **架构** | 带决策节点的循环图 |
| **延迟** | 动态变化，取决于调用次数 |
| **控制力** | 中等，依赖 Agent 决策质量 |
| **灵活性** | 高，可处理复杂推理 |
| **适用场景** | 需调用多源工具的研究助手 |

**示例**：

```python
# Agentic RAG 伪代码
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """搜索内部文档知识库。"""
    docs = retriever.invoke(query)
    return "\n".join(doc.content for doc in docs)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search_docs],
    system_prompt="""你是研究助手。当需要外部信息时，使用 search_docs 工具。
你必须查阅文档后再回答问题。如果工具返回的信息不足，请尝试不同的查询。"""
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我们公司的请假政策是什么？"}]
})
```

### 5.4 Hybrid RAG（混合检索）

**流程**：引入查询增强、检索相关性验证及生成答案质量检查的闭环策略。

```
用户提问
  │
  ▼
┌──────────────┐
│ 查询增强      │  重写/扩写查询
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 检索文档      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 充分性验证    │  检索到的信息足够吗？
└──────┬───────┘
       │
  ┌────┴────┐
  │ 不足    │ 充足
  ▼         ▼
增强查询    ┌──────────────┐
重新检索    │ 生成答案      │
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │ 质量检查      │  答案准确完整吗？
           └──────┬───────┘
                  │
             ┌────┴────┐
             │ 不合格  │ 合格
             ▼         ▼
           重新生成   返回用户
```

**特点**：

| 特性 | 说明 |
|------|------|
| **架构** | 复杂迭代图，多路径分支反馈 |
| **延迟** | 较高，多轮验证 |
| **控制力** | 强，每步都有验证 |
| **灵活性** | 中等，有固定验证流程 |
| **准确度** | 高，迭代验证提升质量 |
| **适用场景** | 领域特定问答、需严格质量控制的场景 |

---

## 六、如何在 LangChain 中实现 Retrieval

### 6.1 方式一：从零构建知识库

使用 LangChain 的文档加载器、文本分割器和向量存储组件，将自有数据构建成可搜索的知识库。

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 步骤 1：加载文档
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()

# 步骤 2：分割文本
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 每块 500 字符
    chunk_overlap=50   # 重叠 50 字符，保持上下文连贯
)
chunks = splitter.split_documents(documents)

# 步骤 3：嵌入并存入向量库
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 步骤 4：创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 返回最相关的 3 个文档块

# 步骤 5：使用检索器
docs = retriever.invoke("公司的年假政策是什么？")
for doc in docs:
    print(f"来源：{doc.metadata['source']}")
    print(f"内容：{doc.page_content[:100]}...")
```

### 6.2 方式二：集成现有系统

若已有 SQL、CRM 或内部文档系统，**无需重建**。可直接将其封装为 Agent 的工具，或查询后作为上下文注入 2-Step RAG 链。

```python
from langchain_core.tools import tool
from langchain.agents import create_agent

# 封装现有系统为工具
@tool
def query_crm(customer_id: str) -> str:
    """查询 CRM 系统中的客户信息。"""
    # 调用现有 CRM API
    data = crm_api.get_customer(customer_id)
    return f"客户：{data['name']}, 等级：{data['tier']}, 订单数：{data['orders']}"

@tool
def search_internal_docs(query: str) -> str:
    """搜索内部文档系统。"""
    # 调用现有文档搜索 API
    results = docs_api.search(query)
    return "\n".join(f"标题：{r['title']}\n内容：{r['snippet']}" for r in results)

# 创建 Agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[query_crm, search_internal_docs],
    system_prompt="""你是企业助手。查询客户信息使用 query_crm，
搜索文档使用 search_internal_docs。
你必须先查阅相关信息再回答用户问题。"""
)
```

### 6.3 方式三：URL 检索工具

```python
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def fetch_url(url: str) -> str:
    """获取指定 URL 的网页内容。
    
    Args:
        url: 要抓取的网页地址
    """
    # 实际实现会调用 HTTP 请求
    return f"网页内容：..."

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[fetch_url],
    system_prompt="""你是研究助手。当需要网上信息时，使用 fetch_url 工具获取网页内容。
你必须先阅读获取到的内容再回答用户问题，并引用来源。"""
)
```

---

## 七、使用场景与最佳实践

### 7.1 场景匹配指南

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| **FAQ 问答** | 2-Step RAG | 延迟低且可预测，控制力强 |
| **文档客服** | 2-Step RAG | 问题标准，检索即可回答 |
| **研究助手** | Agentic RAG | 需调用多源工具，灵活推理 |
| **专业领域问答** | Hybrid RAG | 需要严格质量控制 |
| **模糊查询处理** | Hybrid RAG | 需查询增强和迭代验证 |

### 7.2 最佳实践

#### 实践 1：组件高度模块化

LangChain 组件可随时替换而不破坏核心逻辑。

```python
# 随时可以替换的组件：

# 文档加载器：PDF → Word → Notion
loader = PyPDFLoader("file.pdf")        # 可替换为
loader = Docx2txtLoader("file.docx")    # 或
loader = NotionLoader()

# 文本分割器：递归 → 语义
splitter = RecursiveCharacterTextSplitter()  # 可替换为
splitter = SemanticChunker(embeddings)

# 向量数据库：Chroma → Pinecone → FAISS
vectorstore = Chroma.from_documents(...)     # 可替换为
vectorstore = Pinecone.from_documents(...)   # 或
vectorstore = FAISS.from_documents(...)

# 嵌入模型：OpenAI → Cohere → 开源
embeddings = OpenAIEmbeddings()         # 可替换为
embeddings = CohereEmbeddings()
```

#### 实践 2：为外部检索工具设置访问限制

```python
# 安全实践：域名白名单
ALLOWED_DOMAINS = ["docs.company.com", "wiki.company.com"]

@tool
def fetch_url(url: str) -> str:
    """获取网页内容（仅限公司内部网站）。"""
    from urllib.parse import urlparse
    
    domain = urlparse(url).netloc
    if domain not in ALLOWED_DOMAINS:
        return f"错误：不允许访问 {domain}。允许的域名：{ALLOWED_DOMAINS}"
    
    return fetch_content(url)
```

#### 实践 3：设置异常回退机制

```python
@tool
def search_docs(query: str) -> str:
    """搜索文档，失败时提供回退。"""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关文档。建议联系人工客服或查阅员工手册。"
        return "\n".join(doc.content for doc in docs)
    except Exception as e:
        return f"搜索服务暂时不可用：{str(e)}。根据我的经验，..."
```

#### 实践 4：系统提示词中明确检索要求

```python
system_prompt = """你是专业客服助手。

检索规则：
1. 你必须先查阅知识库再回答用户问题
2. 不要依赖你自己的记忆回答事实性问题
3. 如果知识库中没有相关信息，明确告知用户
4. 回答时引用信息来源，格式为 [来源：文档名称]

如果搜索失败：
- 告知用户搜索服务暂时不可用
- 提供你基于一般知识的参考建议
- 明确标注这是参考建议而非官方答案
"""
```

### 7.3 延迟特性分析

```
┌─────────────────────────────────────────────────────────┐
│              三种模式的延迟特性                            │
│                                                         │
│  2-Step RAG:                                            │
│  ├── LLM 调用次数：固定（1-2 次）                         │
│  ├── 推理延迟：高度可预测                                 │
│  └── 端到端延迟：受检索环节影响（API/网络/DB）             │
│                                                         │
│  Agentic RAG:                                           │
│  ├── LLM 调用次数：动态（1-N 次）                         │
│  ├── 推理延迟：动态变化                                   │
│  └── 端到端延迟：取决于 Agent 决策路径                    │
│                                                         │
│  Hybrid RAG:                                            │
│  ├── LLM 调用次数：动态（多轮验证）                        │
│  ├── 推理延迟：较高                                      │
│  └── 端到端延迟：取决于验证轮次                          │
└─────────────────────────────────────────────────────────┘
```

---

## 八、重要技术细节

### 8.1 分割策略对检索质量的影响

```
分割太小：
  ❌ 丢失上下文，片段信息不完整
  ❌ 检索结果碎片化

分割太大：
  ❌ 包含过多无关信息
  ❌ 向量表示被稀释，检索精准度下降

推荐策略：
  ✅ RecursiveCharacterTextSplitter（递归字符分割）
     - 尝试多种分隔符（\n\n, \n, 空格）
     - 保持语义完整性
     - 可配置 chunk_size 和 chunk_overlap

  ✅ 典型配置：
     - chunk_size: 500-1000 字符
     - chunk_overlap: 50-100 字符（保持上下文连贯）
```

### 8.2 检索器参数调优

```python
# 控制返回文档数量
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 控制相似度阈值（只返回足够相关的文档）
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5}
)

# 最大边际相关性（MMR）—— 平衡相关性和多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.5}
)
# lambda_mult: 1 = 只相关性，0 = 只多样性
```

### 8.3 安全与容错

Agentic 实现中需通过提示词强制约束：

```python
system_prompt = """检索规则：
1. 必须查阅文档后再回答问题
2. 如果工具返回的信息不足以回答问题，请尝试不同的查询
3. 如果多次搜索仍无结果，告知用户信息不足
4. 失败时提供基于你的专业经验的参考建议，但需明确标注"""
```

---

## 九、常见错误与陷阱

### 陷阱 1：整个文档不分割直接嵌入

**现象**：检索结果包含大量无关信息，LLM 难以聚焦。

```python
# ❌ 错误：整篇文档作为一个块
docs = loader.load()
vectorstore = Chroma.from_documents(docs, embeddings)
# 每个文档可能几千到上万字符，检索时返回整篇！

# ✅ 正确：分割为小块
chunks = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(chunks, embeddings)
# 每块 500 字符，检索时返回最相关的片段
```

### 陷阱 2：忽略 chunk_overlap

**现象**：分割处丢失上下文，检索结果不完整。

```python
# ❌ 错误：无重叠
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# ✅ 正确：设置重叠
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# 相邻块有 50 字符重叠，保持上下文连贯
```

### 陷阱 3：检索结果不验证直接使用

**现象**：检索到不相关文档，LLM 基于错误信息回答。

```python
# ❌ 错误：不验证直接使用
docs = retriever.invoke(query)
context = "\n".join(doc.content for doc in docs)
# 即使 docs 为空或不相关，也传入

# ✅ 正确：验证后使用
docs = retriever.invoke(query)
if not docs:
    return "未找到相关信息，无法回答。"
context = "\n".join(doc.content for doc in docs)
```

### 陷阱 4：Agentic RAG 中未约束 Agent 行为

**现象**：Agent 不调用检索工具，直接凭记忆回答。

```python
# ❌ 错误：提示词未约束
system_prompt = "你是客服助手。"
# Agent 可能直接回答，不查阅文档

# ✅ 正确：明确约束
system_prompt = """你是客服助手。
你必须先查阅知识库再回答用户问题。
不要依赖你自己的记忆回答事实性问题。
如果知识库中没有相关信息，明确告知用户。"""
```

---

## 十、总结：Retrieval 的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Retrieval** | 开卷考试 | 动态获取外部知识 |
| **Document Loader** | 图书管理员 | 从各种来源收集资料 |
| **Text Splitter** | 裁纸刀 | 将大文档切分为小片段 |
| **Embedding** | 翻译官 | 将文本转为向量 |
| **Vector Store** | 图书馆 | 存储和检索向量 |
| **Retriever** | 搜索引擎 | 接收查询返回相关文档 |
| **2-Step RAG** | 固定流水线 | 先检索后生成 |
| **Agentic RAG** | 研究助手 | 智能体自主决定何时检索 |
| **Hybrid RAG** | 质量控制员 | 多轮验证确保答案质量 |

**Retrieval 的本质**是**让 LLM 从"闭卷考试"变为"开卷考试"**。它通过文档加载、分割、嵌入、存储、检索这一完整管线，让模型能够在运行时动态获取相关的外部知识，从而克服上下文有限和知识静态两大固有局限，实现更准确、更实时的回答。

```
一句话总结：

Retrieval 是 RAG 的基础架构，核心是"将外部数据转化为
可搜索的知识库，在查询时动态检索相关上下文注入 LLM"，
弥补模型内部知识的不足，减少幻觉，提升回答准确度。
```
