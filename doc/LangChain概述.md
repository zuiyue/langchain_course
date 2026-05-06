# LangChain 概述 — 构建 AI 应用的开发框架

## 一、LangChain 是什么

LangChain 是一个开源开发框架，用于构建基于大语言模型（LLM）的应用程序。它的核心定位可以用一句话概括：**让开发者用代码的方式，把 LLM 的能力可靠地落地到生产环境中。**

大语言模型本身只是一个"文本生成器"——给它输入，它返回输出。但真实的业务场景需要更多：连接数据库、搜索网页、读写文件、保持对话记忆、控制执行流程、保证输出格式……LangChain 提供的就是这些"最后一公里"的基础设施。

## 二、为什么需要 LangChain

直接调用模型 API 会遇到几个根本性问题：

1. **模型是"失忆"的**——每次调用都是独立的，不会记住之前的对话
2. **模型是"离线"的**——训练数据有截止日期，无法获取最新信息
3. **模型是"孤立"的**——无法直接操作系统、数据库或外部 API
4. **输出是"不可控"的**——返回自由文本，难以直接对接业务逻辑

LangChain 逐一解决了这些问题：

| 问题 | LangChain 方案 |
|------|---------------|
| 失忆 | Checkpointer（检查点持久化） |
| 离线 | Retrieval（检索增强，RAG） |
| 孤立 | Tools（工具调用） |
| 不可控 | Structured Output（结构化输出） |

## 三、核心概念

### 3.1 模型（Models）

LangChain 通过统一的接口对接多家厂商的模型，切换模型只需改一行配置。本教程以 DeepSeek 为例：

```python
# DeepSeek 

from langchain.chat_models import init_chat_model
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_model = init_chat_model(
    "deepseek:deepseek-reasoner",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

```


### 3.2 工具（Tools）

工具是模型与外部世界交互的桥梁。每个工具是一个函数，带有名称、描述和参数 schema——模型根据描述**自主决定**何时调用：

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name
    """
    return f"Weather in {location}: Sunny, 72F"
```

> **关键细节**：工具描述必须清晰具体。模型靠描述来判断何时调用，描述模糊会导致工具不被使用或被误用。

### 3.3 Agent（智能体）

Agent 是 LangChain 的核心抽象。通过 `create_agent()` 创建，它自动处理"理解用户 → 选择工具 → 执行 → 整合结果 → 回复"的循环：

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

model = init_chat_model(
    "deepseek:deepseek-reasoner",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name
    """
    return f"Weather in {location}: Sunny, 72F"


agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}]
})
print(result["messages"][-1].content)
```

> **注意**：DeepSeek 的 `deepseek-chat` 模型支持工具调用（function calling），这是 Agent 能够自主调用工具的前提。如果你的模型不支持 function calling，Agent 模式将无法工作。

### 3.4 记忆（Persistence）

Agent 默认是无状态的。加上 `checkpointer` 后，对话可以跨轮次持续：

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from langchain.chat_models import init_chat_model
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from langchain_core.tools import tool

model = init_chat_model(
    "deepseek:deepseek-reasoner",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name
    """
    return f"Weather in {location}: Sunny, 72F"


agent = create_agent(
    model=model,
    tools=[get_weather],
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "session-1"}}
agent.invoke({"messages": [{"role": "user", "content": "I'm Bob"}]}, config=config)
result = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config=config)
# 输出: "Your name is Bob"
```

`thread_id` 是对话的唯一标识。同一个 `thread_id` 下的多轮调用共享上下文，不同 `thread_id` 之间互不干扰。

### 3.5 结构化输出（Structured Output）

业务系统需要的是结构化数据而非自由文本。LangChain 支持用 Pydantic 模型约束输出格式：

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="your-deepseek-api-key",
    openai_api_base="https://api.deepseek.com",
)

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str = Field(description="Phone number with area code")

# 方式一：Agent 级别的结构化输出
agent = create_agent(
    model=model,
    tools=[],
    response_format=ContactInfo,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info: John, john@example.com, 555-1234"}]
})
print(result["structured_response"])
# ContactInfo(name='John', email='john@example.com', phone='555-1234')

# 方式二：模型级别的结构化输出（不需要 Agent）
structured_model = model.with_structured_output(ContactInfo)
response = structured_model.invoke("Extract: John, john@example.com, 555-1234")
print(response)
# ContactInfo(name='John', email='john@example.com', phone='555-1234')
```

> **注意**：结构化输出依赖模型的原生支持。DeepSeek 的 `deepseek-chat` 模型支持 function calling，因此 `with_structured_output` 可以正常工作。如果遇到兼容性问题，可以尝试使用 `method="json_mode"` 参数。

### 3.6 中间件（Middleware）

中间件拦截 Agent 的执行流程，在不修改核心逻辑的前提下增加能力：

- **HumanInTheLoopMiddleware** — 在敏感操作前暂停，等待人工审批
- **自定义中间件** — 通过 `@wrap_tool_call` 装饰器实现日志、重试、限流等

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="your-deepseek-api-key",
    openai_api_base="https://api.deepseek.com",
)

agent = create_agent(
    model=model,
    tools=[dangerous_tool],
    checkpointer=MemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"dangerous_tool": True})],
)

# 执行到 dangerous_tool 时会暂停，等待人工审批后恢复
config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Run the dangerous task"}]},
    config=config,
)
```

> **注意**：`HumanInTheLoopMiddleware` 必须配合 `checkpointer` 使用，因为中断后恢复需要持久化状态。

## 四、LangChain 在整个生态中的位置

LangChain、LangGraph、Deep Agents 是**分层关系**，不是竞争关系：

```
┌─────────────────────────────────────────┐
│              Deep Agents                │  ← 最高层：开箱即用，适合长期任务
│   (规划、记忆、技能、文件管理)            │
├─────────────────────────────────────────┤
│               LangGraph                 │  ← 编排层：图结构、循环、分支
│    (节点、边、状态、持久化)               │
├─────────────────────────────────────────┤
│               LangChain                 │  ← 基础层：模型、工具、链、RAG
│      (models, tools, prompts, RAG)      │
└─────────────────────────────────────────┘
```

大多数入门场景，LangChain 本身就够了——单个 Agent 调用工具、返回结果。当需要复杂控制流（循环、分支、并行）时，再向上引入 LangGraph；当需要自动规划、子任务委派、持久记忆时，再引入 Deep Agents。

## 五、常见误区

| 误区 | 正确做法 |
|------|---------|
| 不加 checkpointer，Agent 每轮"失忆" | 配置 MemorySaver + thread_id |
| 工具描述模糊或缺失 | 写清楚何时使用 + Args 说明 |
| 不设 recursion_limit，可能死循环 | invoke 时设置 `config={"recursion_limit": 10}` |
| 直接访问 `result.content` | 正确访问 `result["messages"][-1].content` |
| 使用不支持 function calling 的模型 | 确认模型支持工具调用，如 deepseek-chat |


## 六、总结

LangChain 的本质是**LLM 应用的工程化框架**。它把模型调用、工具集成、对话记忆、输出约束这些生产环境中必须解决的问题封装成一致的 API，让开发者专注于业务逻辑而非底层对接。

从一条模型调用链，到一个带工具的 Agent，再到完整的 RAG 流水线——LangChain 提供了一条从原型到生产的渐进路径。



