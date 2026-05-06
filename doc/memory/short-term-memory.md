# LangChain Short-term Memory 核心概念教程


---

## 一、什么是短期记忆？

### 1.1 从人的记忆到 AI 的记忆

人的记忆分为**短期记忆**和**长期记忆**：

```
人的短期记忆：
  "刚才你说你叫小明"  ← 记住最近说的话
  "我们正在讨论天气"  ← 记住当前话题

人的长期记忆：
  "小明的生日是 3 月 15 日"  ← 跨时间保留的重要信息
  "小明喜欢喝咖啡"          ← 积累的知识
```

AI 模型也有类似的概念：

**短期记忆（Short-term Memory）** 允许应用在**单个线程（Thread）或会话内**记住之前的交互信息。最常见的形式是**对话历史（Conversation History）**。

```
没有短期记忆：
  用户："我叫小明"
  AI："好的，小明！"
  
  （下一条消息）
  用户："我叫什么名字？"
  AI："我不知道。"  ← 忘了！

有短期记忆：
  用户："我叫小明"
  AI："好的，小明！"
  [保存：对话历史包含两条消息]
  
  （下一条消息，使用同一个 thread_id）
  用户："我叫什么名字？"
  AI："你叫小明！"  ← 记住了！
```

### 1.2 短期记忆 vs 长期记忆

| 特性 | 短期记忆 | 长期记忆 |
|------|----------|----------|
| **范围** | 单个会话内 | 跨多个会话 |
| **内容** | 完整的对话历史 | 提炼的关键信息 |
| **存储** | Checkpointer（内存/数据库） | Store（键值存储） |
| **生命周期** | 会话结束后可丢弃 | 持久保留 |
| **类比** | 人的工作记忆 | 人的知识库 |

**核心定义**：短期记忆是**维持当前对话连贯性的机制**。它不跨会话保留，但确保在单次会话中，模型能够基于上下文做出连贯的响应。

---

## 二、短期记忆的工作原理

### 2.1 状态管理模型

短期记忆的实现基于一个核心概念：**State（状态）**。

```
┌─────────────────────────────────────────────────────────┐
│                    状态管理流程                           │
│                                                         │
│   用户输入 ──► 更新 State                                │
│                    │                                     │
│                    ▼                                     │
│              ┌──────────┐                                │
│              │  State   │  messages: [msg1, msg2, ...]   │
│              │         │  user_id: "123"                 │
│              │         │  preferences: {...}             │
│              └────┬────┘                                │
│                   │                                      │
│                   ▼                                      │
│              读取 State ──► 模型推理 ──► 输出             │
│                                                         │
│   每次调用或步骤完成后，State 通过 Checkpointer 持久化     │
└─────────────────────────────────────────────────────────┘
```

**关键点**：

- Agent 将短期记忆作为其 **State 的一部分**进行管理
- 默认通过 `messages` 键存储对话历史
- 状态存储在图（Graph）中，确保不同 `thread_id` 的上下文**相互隔离**
- 每次调用 Agent 或完成一个步骤（如工具调用）时**更新状态**
- 每个步骤开始时**读取最新状态**

### 2.2 Checkpointer —— 记忆的"保存点"

**Checkpointer（检查点）** 是实现短期记忆持久化的核心机制。它会在每次状态更新时保存快照，支持随时中断并恢复线程。

**开发/测试环境**：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()  # 内存存储，重启后丢失
```

**生产环境**：

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(connection_string="postgresql://...")  # 数据库持久化
```

**常见的 Checkpointer 选项**：

| 类型 | 适用场景 | 持久化 | 安装依赖 |
|------|----------|--------|----------|
| `InMemorySaver` | 开发/测试 | ❌ 重启丢失 | 无 |
| `PostgresSaver` | 生产环境 | ✅ | `langgraph-checkpoint-postgres` |
| `SqliteSaver` | 轻量级生产 | ✅ | `langgraph-checkpoint-sqlite` |
| `AzureCosmosDBSaver` | 云环境 | ✅ | `langgraph-checkpoint-azure-cosmosdb` |

### 2.3 会话隔离：thread_id 的作用

```
线程 A（thread_id: "user-1"）:
  用户 A："我叫小明"
  ──► State A: {"messages": [HumanMessage("我叫小明")]}

线程 B（thread_id: "user-2"）:
  用户 B："我叫小红"
  ──► State B: {"messages": [HumanMessage("我叫小红")]}

两个线程的状态完全隔离，互不干扰！
```

**实现方式**：

```python
config = {"configurable": {"thread_id": "user-1"}}

# 调用时传入 config
result = agent.invoke({"messages": [HumanMessage("你好")]}, config=config)
```

---

## 三、自定义 State —— 扩展短期记忆

### 3.1 默认的 AgentState

默认情况下，Agent 使用 `AgentState`，它主要包含：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史
```

### 3.2 添加自定义字段

你可以继承 `AgentState`，添加自定义字段来存储额外的短期状态：

```python
from typing import TypedDict, Annotated
from langchain.agents import AgentState
from langgraph.graph.message import add_messages

class CustomState(AgentState):
    # 添加用户 ID
    user_id: str
    
    # 添加用户偏好
    preferences: dict
    
    # 添加临时标记
    current_task: str

# 创建 Agent 时传入自定义 State
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search],
    state_schema=CustomState,
)
```

### 3.3 在工具中读写状态

**读取状态**：

```python
from langchain.tools import tool
from langchain_core.tools import ToolRuntime

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """获取当前用户的信息。"""
    # 读取短期状态
    state = runtime.state
    user_id = state.get("user_id", "未知")
    return f"当前用户 ID: {user_id}"
```

**更新状态**：

```python
from langgraph.types import Command

@tool
def set_preference(pref: str, runtime: ToolRuntime) -> Command:
    """设置用户的偏好。"""
    return Command(
        update={"preferences": {"language": pref}},
        graph=Command.PARENT
    )
```

---

## 四、上下文窗口管理

### 4.1 为什么需要管理？

LLM 的上下文窗口是**有限的**，而且：

- **长上下文 = 高成本**：更多的 token 意味着更高的费用
- **长上下文 = 性能下降**：模型在超长上下文中查找信息的能力会下降
- **无限增长不可行**：对话持续进行时，消息列表会无限增长，最终超出窗口限制

```
对话进行 100 轮后：
  messages: [msg1, msg2, msg3, ..., msg200]  ← 200 条消息！
  
  问题：
  1. 超出模型的上下文窗口（如 128K token）
  2. 成本急剧上升
  3. 模型在长文本中"迷失"，回答质量下降
```

### 4.2 管理策略一：Trim Messages（裁剪）

保留最早和最近的一定数量消息，**丢弃中间部分**。

```
原始消息列表：
  [System, User1, AI1, User2, AI2, User3, AI3, ..., User50, AI50]

裁剪后（保留首尾各 5 条）：
  [System, User1, AI1, User2, AI2, User46, AI46, User47, AI47, User48, AI48, User49, AI49, User50, AI50]
```

**通过 `@before_model` 中间件实现**：

```python
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state):
    """裁剪消息列表，保留首尾。"""
    messages = state["messages"]
    
    if len(messages) <= 10:
        return state  # 不超过阈值，不裁剪
    
    # 保留前 3 条和最后 7 条
    trimmed = messages[:3] + messages[-7:]
    return {"messages": trimmed}
```

### 4.3 管理策略二：Delete Messages（删除）

从 LangGraph 状态中**永久移除**特定或全部消息。

```python
from langchain_core.messages import RemoveMessage

# 删除指定的消息
state["messages"] = [
    msg for msg in state["messages"]
    if not (hasattr(msg, 'content') and "临时信息" in msg.content)
]

# 或使用 RemoveMessage（推荐，确保 Reducer 正确处理）
from langgraph.types import Command

Command(update={
    "messages": [RemoveMessage(id=msg_to_delete.id)]
})
```

**⚠️ 重要约束**：

使用 `RemoveMessage` 需确保状态键配置了 `add_messages` reducer（默认 `AgentState` 已内置支持）。

### 4.4 管理策略三：Summarize Messages（总结）

用 LLM 总结早期历史并**替换原文**，保留关键信息。

```
原始消息：
  [User1: "我们讨论了项目需求...", 
   AI1: "我理解了，你需要...",
   User2: "还有技术架构方面...",
   AI2: "建议使用微服务..."]

总结后：
  [AIMessage("用户讨论了项目需求和技术架构，\
              决定采用微服务方案。"),
   User3: "好的，那我们开始吧"]
```

**使用内置 `SummarizationMiddleware`**：

```python
from langchain.agents.middleware import SummarizationMiddleware

# 配置：当消息超过 50 个 token 时触发总结，保留最近 5 条消息
summarizer = SummarizationMiddleware(
    model="anthropic:claude-sonnet-4-5",
    max_tokens_before_summary=50,
    max_messages_to_keep=5
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search],
    middleware=[summarizer],
)
```

**总结流程示意图**：

```
┌─────────────────────────────────────────────────────────┐
│                  消息总结流程                              │
│                                                         │
│   消息历史超过阈值？                                      │
│         │                                               │
│         ├── 否 ──► 保持原样                              │
│         │                                               │
│         └── 是                                          │
│               │                                         │
│               ▼                                         │
│         ┌──────────┐                                    │
│         │  LLM 总结 │  总结早期历史为一段简洁的概述        │
│         └────┬────┘                                    │
│              │                                          │
│              ▼                                          │
│         ┌──────────┐                                    │
│         │  替换消息 │  用总结替换原文，保留最近几条消息     │
│         └────┬────┘                                    │
│              │                                          │
│              ▼                                          │
│         精简后的消息历史 ──► 继续对话                     │
└─────────────────────────────────────────────────────────┘
```

### 4.5 管理策略四：自定义策略

你可以通过中间件实现任何自定义的消息管理策略：

```python
from langchain.agents.middleware import before_model

@before_model
def filter_important_messages(state):
    """只保留标记为重要的消息。"""
    messages = state["messages"]
    
    important_messages = [
        msg for msg in messages
        if msg.metadata.get("important", False) or 
           isinstance(msg, SystemMessage)
    ]
    
    return {"messages": important_messages}
```

---

## 五、中间件在短期记忆中的应用

### 5.1 `@before_model` —— 模型推理前处理

在模型推理**之前**拦截消息，适合做裁剪、过滤、注入动态提示词。

```
流程：__start__ → before_model → model → (tools 或 __end__)
                                      tools 执行后返回 before_model 循环
```

```python
from langchain.agents.middleware import before_model

@before_model
def inject_dynamic_prompt(state):
    """在模型推理前注入动态系统提示。"""
    # 根据用户身份生成不同的提示
    user_role = state.get("user_role", "default")
    
    if user_role == "admin":
        dynamic_prompt = SystemMessage("你是管理员助手，可以访问所有数据。")
    else:
        dynamic_prompt = SystemMessage("你是普通用户助手。")
    
    return {
        "messages": [dynamic_prompt] + state["messages"]
    }
```

### 5.2 `@after_model` —— 模型推理后处理

在模型推理**之后**拦截输出，适合做验证、清理、日志记录。

```
流程：__start__ → model → after_model → (tools 或 __end__)
                                    tools 执行后返回 model 循环
```

```python
from langchain.agents.middleware import after_model

@after_model
def validate_output(state):
    """验证模型输出是否符合要求。"""
    last_message = state["messages"][-1]
    
    # 检查是否包含敏感信息
    if "password" in last_message.content.lower():
        return {
            "messages": [
                {"role": "assistant", "content": "抱歉，我无法提供密码信息。"}
            ]
        }
    
    return state  # 验证通过，保持原样
```

---

## 六、最佳实践与设计哲学

### 6.1 生产环境务必使用数据库 Checkpointer

```python
# ❌ 开发环境：内存存储
checkpointer = InMemorySaver()

# ✅ 生产环境：数据库持久化
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(
    connection_string="postgresql://user:pass@host:5432/db"
)
```

**原因**：
- 服务重启后记忆不丢失
- 支持多实例共享状态
- 便于审计和调试

### 6.2 删除/裁剪消息时必须保证格式正确

**⚠️ 重要警告**：删除或裁剪消息时，**必须保证剩余历史符合 LLM 提供商的格式要求**。

常见格式要求：

```
✅ 正确：
  [HumanMessage, AIMessage, HumanMessage]  ← 以 user 消息开头
  [AIMessage(tool_calls=[...]), ToolMessage, AIMessage]  ← 工具调用成对

❌ 错误：
  [AIMessage]  ← 没有以 user 消息开头！
  [AIMessage(tool_calls=[...]), HumanMessage]  ← 缺少 ToolMessage！
```


### 6.3 根据场景选择合适的管理策略

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 对话过长，但需要保留上下文 | Summarize（总结） | 保留关键信息 |
| 只需要最近的对话 | Trim（裁剪） | 简单高效 |
| 有无关或临时消息 | Delete（删除） | 精确清理 |
| 特殊业务逻辑 | 自定义中间件 | 灵活控制 |

---

## 七、总结：短期记忆的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **State（状态）** | 记事本 | 存储当前会话的所有信息 |
| **Checkpointer** | 保存按钮 | 定期保存状态快照 |
| **thread_id** | 笔记本编号 | 隔离不同会话的状态 |
| **Trim** | 撕掉旧页 | 裁剪中间消息 |
| **Delete** | 擦除内容 | 永久移除特定消息 |
| **Summarize** | 做笔记 | 用总结替代原文 |
| **Middleware** | 过滤器 | 在推理前后处理消息 |

**短期记忆的本质**是**维持当前对话的连贯性**。它通过状态管理和检查点机制，让模型能够"记住"之前说过什么，从而做出连贯、有上下文的响应。但它不跨会话保留——那是长期记忆的职责。

---

